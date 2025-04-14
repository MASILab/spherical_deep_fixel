import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from .dataset import RandomMeshDataset, GeneratedMeshDataset
from .models import CrossingFiberMeshMLP
from .utils import plot_odf, plot_mesh, pdf2odfs, match_odfs, angular_corr_coeff
from pathlib import Path
import matplotlib.pyplot as plt
from dipy.reconst.shm import sf_to_sh, convert_sh_descoteaux_tournier, gen_dirac, sph_harm_ind_list
import pandas as pd
    
def train_mesh_model(
    run_name,
    lr=1e-3,
    batch_size=512,
    n_fibers=2,
    n_steps=10000,
    validation_patience=5,
    loss_name="MSE",
    model="mesh_mlp",
    gpu_id=0,
    seed=None,
    mesh_subdivide=3,
    kappa=100,
    save_dir="./models",
    healpix=False
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize wandb
    save_dir = Path(save_dir) / run_name
    if not save_dir.exists():
        save_dir.mkdir()

    config = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "n_fibers": n_fibers,
        "n_steps": n_steps,
        "validation_patience": validation_patience,
        "loss": loss_name,
        "seed": seed,
        "mesh_subdivide": mesh_subdivide,
        "kappa": kappa,
        "model": model,
        "healpix": healpix,
    }

    # Set up Weights and Biases
    wandb.login()
    run = wandb.init(project="deepfixel", name=run_name, config=config)

    # Set up datasets
    train_dataset = RandomMeshDataset(n_fibers=n_fibers, l_max=6, seed=seed, subdivide=mesh_subdivide, kappa=kappa, healpix=healpix)
    val_dataset = RandomMeshDataset(
        n_fibers=n_fibers, l_max=6, seed=seed + 1, size=1000, deterministic=True, subdivide=mesh_subdivide, kappa=kappa, healpix=healpix
    )

    # Set up dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_mesh = train_dataset.n_mesh
    sphere = train_dataset.sphere
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f'Training on device {device}...')

    # Set up model
    if model == "mesh_mlp":
        model = CrossingFiberMeshMLP(n_mesh=n_mesh, device=device, sphere=sphere)
    else:
        raise ValueError(f"Model {model} not recognized")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_name == "MSE":
        criterion = torch.nn.MSELoss()
    elif loss_name == 'L1':
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError(f"Loss {loss_name} not recognized")

    print(model)
    pdf_mesh, total_odf_mesh = next(iter(train_loader))
    print(f"Input shape: {total_odf_mesh.shape}")
    print(f"Output shape: {model(total_odf_mesh).shape}")
    print(model(total_odf_mesh))

    # Train model
    model.to(device)
    best_val_loss = np.inf
    step = 1
    patience = 0

    model.train()
    for step, data in tqdm(enumerate(train_loader), total=n_steps):
        pdf_mesh, total_odf_mesh = data
        pdf_mesh = pdf_mesh.to(device)
        total_odf_mesh = total_odf_mesh.to(device)

        # Update model
        optimizer.zero_grad()
        output = model(total_odf_mesh)

        loss = criterion(output, pdf_mesh)
        loss.backward()
        optimizer.step()

        # Log to wandb
        wandb.log({"train_loss": loss.item()}, step=step)
        tqdm.write(f"Step {step}, Training loss: {loss.item()}")

        # Validate model at intervals of 50 steps
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_data in tqdm(val_loader, total=len(val_loader)):
                    pdf_mesh, total_odf_mesh = val_data
                    pdf_mesh = pdf_mesh.to(device)
                    total_odf_mesh = total_odf_mesh.to(device)

                    output = model(total_odf_mesh)
                    val_loss += criterion(output, pdf_mesh).item()
                val_loss /= len(val_loader)
                wandb.log({"val_loss": val_loss}, step=step)
                tqdm.write(f"Step {step}, Validation loss: {val_loss}")

                # Try plotting a few outputs
                fig, ax = plt.subplots(
                    3, 4, figsize=(15, 15), subplot_kw={"projection": "3d"}
                )
                pdf_mesh, total_odf_mesh = next(iter(val_loader))
                pdf_mesh = pdf_mesh.numpy()

                output = model(total_odf_mesh.to(device))
                output = output.cpu().numpy()
                total_odf_mesh = total_odf_mesh.cpu().numpy()

                for i in range(3):
                    pdf_mesh_to_plot = pdf_mesh[i]
                    output_to_plot = output[i]
                    total_odf_mesh_to_plot = total_odf_mesh[i]

                    # Fit total ODF mesh to SH
                    total_odf_sh = sf_to_sh(total_odf_mesh_to_plot, sphere=sphere, sh_order_max=6, basis_type='tournier07')
                    plot_odf(total_odf_sh, ax=ax[i, 0], color="r")
                    plot_mesh(total_odf_mesh_to_plot, sphere=sphere, ax=ax[i, 1])
                    plot_mesh(output_to_plot, sphere=sphere, ax=ax[i, 2])
                    plot_mesh(pdf_mesh_to_plot, sphere=sphere, ax=ax[i, 3])


                ax[0, 0].set_title("Total ODF (SH)")
                ax[0, 1].set_title("Total ODF (Mesh)")
                ax[0, 2].set_title("Predicted PDF (Mesh)")
                ax[0, 3].set_title("True PDF (Mesh)")

                fig.suptitle(f"Validation (Step {step})")

                img_path = save_dir / f"mesh_plots_{step}.png"
                fig.savefig(img_path)
                plt.close(fig)

                wandb.log({"mesh_plots": wandb.Image(str(img_path))}, step=step)

            if val_loss < best_val_loss:
                wandb.run.summary["best_val_loss"] = val_loss
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), save_dir / f"best_model.pth")
            else:
                patience += 1
                if patience > validation_patience:
                    tqdm.write(
                        f"Validation loss has not improved in {validation_patience} steps. Breaking."
                    )
                    break

            model.train()

        # Break if we reach n_steps
        if step > n_steps:
            break

def test_mesh_model(
    model_path,
    n_fibers,
    subdivide_mesh,
    amp_threshold,
    output_dir,
    kappa=100,
    batch_size=512,
    test_dir="./test_data",
    gpu_id=0,
    healpix=False,
):
    # Load data
    test_dataset = GeneratedMeshDataset(n_fibers=n_fibers, directory=test_dir, return_fixels=True, subdivide=subdivide_mesh, kappa=kappa, healpix=healpix)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_mesh = test_dataset.n_mesh
    sphere = test_dataset.icosphere
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = CrossingFiberMeshMLP(n_mesh=n_mesh)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)

    m_list, l_list = sph_harm_ind_list(6)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    test_results = {
        "voxel_index": [],
        "fiber_index": [],
        "true_n_fibers": [],
        "est_n_fibers": [],
        "true_theta": [],
        "true_phi": [],
        "true_v": [],
        "est_theta_matched": [],
        "est_phi_matched": [],
        "est_v_matched": [],
        "acc": [],
    }
    with torch.no_grad():
        
        for idx, test_data in enumerate(tqdm(test_loader, desc='test set')):
            pdf_mesh, total_odf_mesh, fixels = test_data
            pdf_mesh = pdf_mesh.to(device)
            total_odf_mesh = total_odf_mesh.to(device)

            output = model(total_odf_mesh)

            # Move back to CPU
            pdf_mesh = pdf_mesh.cpu().numpy()
            output = output.cpu().numpy()
            fixels = fixels.numpy()

            for i in range(len(pdf_mesh)):
                single_pdf_mesh = pdf_mesh[i]
                single_output = output[i]
                theta, phi, vol = fixels[i].T

                # Sort by vol and remove any with vol = 0
                sort_idx = np.argsort(vol)[::-1]
                sort_idx = sort_idx[vol[sort_idx] > 0]
                theta = theta[sort_idx]
                phi = phi[sort_idx]
                vol = vol[sort_idx]

                true_odf = np.array([convert_sh_descoteaux_tournier(gen_dirac(m_list, l_list, theta=t, phi=p))*v for t, p, v in zip(theta, phi, vol)])
                est_odf, est_dirs, est_vol = pdf2odfs(single_output, sphere, amp_threshold=amp_threshold)


                # Match them
                est_odf_matched, index_array = match_odfs(true_odf, est_odf)

                est_n_fibers = len(est_odf_matched)
                true_n_fibers = len(true_odf)

                # Calculate ACC for each fiber
                for j, (odf1, odf2) in enumerate(zip(true_odf, est_odf_matched)):
                    acc = angular_corr_coeff(odf1, odf2)
                    test_results["voxel_index"].append(i)
                    test_results["fiber_index"].append(j)
                    test_results["true_n_fibers"].append(true_n_fibers)
                    test_results["est_n_fibers"].append(est_n_fibers)
                    test_results["true_theta"].append(theta[j])
                    test_results["true_phi"].append(phi[j])
                    test_results["true_v"].append(vol[j])
                    test_results["est_theta_matched"].append(est_dirs[j, 0])
                    test_results["est_phi_matched"].append(est_dirs[j, 1])
                    test_results["est_v_matched"].append(est_vol[j])
                    test_results["acc"].append(acc)

                # if est_n_fibers < true_n_fibers, add dummy rows
                missing_fibers = true_n_fibers - est_n_fibers
                if missing_fibers > 0:
                    for j in range(est_n_fibers, true_n_fibers):
                        test_results["voxel_index"].append(i)
                        test_results["fiber_index"].append(j)
                        test_results["true_n_fibers"].append(true_n_fibers)
                        test_results["est_n_fibers"].append(est_n_fibers)
                        test_results["true_theta"].append(theta[j])
                        test_results["true_phi"].append(phi[j])
                        test_results["true_v"].append(vol[j])
                        test_results["est_theta_matched"].append(np.nan)
                        test_results["est_phi_matched"].append(np.nan)
                        test_results["est_v_matched"].append(np.nan)
                        test_results["acc"].append(0)

            # Plot a few examples
            fig, ax = plt.subplots(3,8, figsize=(15, 15), subplot_kw={"projection": "3d"})
            start_idx=4
            for i in range(3):
                theta, phi, vol = fixels[i+start_idx].T
                # Sort by vol
                sort_idx = np.argsort(vol)[::-1]
                sort_idx = sort_idx[vol[sort_idx] > 0]
                theta = theta[sort_idx]
                phi = phi[sort_idx]
                vol = vol[sort_idx]

                true_odf = np.array([convert_sh_descoteaux_tournier(gen_dirac(m_list, l_list, theta=t, phi=p))*v for t, p, v in zip(theta, phi, vol)])
                est_odf = pdf2odfs(output[i+start_idx], sphere, amp_threshold=amp_threshold)[0]

                # Match them
                est_odf, _ = match_odfs(true_odf, est_odf)

                plot_mesh(pdf_mesh[i+start_idx], sphere, ax=ax[i, 0], cmap="cmc.batlow")
                plot_mesh(output[i+start_idx], sphere, ax=ax[i, 1], cmap="cmc.batlow")

                ax[i, 0].set_title("True PDF")  
                ax[i, 1].set_title("Predicted PDF")

                for j in range(max(len(true_odf), len(est_odf))):
                    if j < len(true_odf):
                        plot_odf(true_odf[j], ax=ax[i, j*2+2], color="b")
                        ax[i, j*2+2].set_title(f"True Fiber")
                    if j < len(est_odf):
                        plot_odf(est_odf[j], ax=ax[i, j*2+3], color="r")
                        if j < len(true_odf):
                            acc = angular_corr_coeff(true_odf[j], est_odf[j])
                            ax[i, j*2+3].set_title(f"Estimated Fiber\nACC: {acc:.2f}")          
                        else:
                            ax[i, j*2+3].set_title(f"Estimated Fiber")     

    # Save results
    test_results = pd.DataFrame(test_results)
    test_results.to_csv(output_dir / "test_results.csv", index=False)

    fig.savefig(output_dir / "test_examples.png")
    plt.close(fig)
