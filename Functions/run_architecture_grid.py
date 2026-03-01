def run_architecture_grid(trainer, hidden_dims_list, latent_dims_list,
                          base_model_params, base_train_params, base_mi_params,
                          file_name="", Debug=False, Validation=False):

    results_grid = {}

    for h_dim, z_dim in product(hidden_dims_list, latent_dims_list):

        exp_name = f"h{h_dim}_z{z_dim}"

        model_params = base_model_params.copy()
        model_params["hiddenDim"] = h_dim
        model_params["latentDim"] = z_dim

        exp = ExperimentConfig(
            name=exp_name,
            model_params=model_params,
            train_params=base_train_params,
            mi_params=base_mi_params
        )

        print(f"\n--- Running experiment {exp_name} ---")
        out = trainer.run(exp, Debug, Validation)
        print("\n*************************************************\n")

        results_grid[exp_name] = {
            "mi_history": out["mi_history"],
            "train_loss": out["model"].train_loss_history,
            "val_loss": out["model"].val_loss_history
        }

    if file_name != "":
        method = base_mi_params['method']
        if isinstance(method, list):
            method_array = "".join(method)
        else: method_array = method
        
        suffix = f"-{base_model_params['bit_type']}-{method_array}-sigma{base_mi_params['sigma']}"
        with open("Results/" + file_name + suffix + ".pkl", "wb") as f:
            pickle.dump(results_grid, f)

    return results_grid