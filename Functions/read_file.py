def read_file(filename, metric="mi_history", unpack=False):

    path = f"Results/{filename}"

    try:
        with open(path, "rb") as f:
            full_results = pickle.load(f)

        if metric == "all":
            extracted_data = full_results
        else:
            extracted_data = {}
            for exp_name, exp_data in full_results.items():
                if metric in exp_data:
                    extracted_data[exp_name] = exp_data[metric]
                else:
                    print(f"Warning: Metric {metric} not found in {exp_name}")

        if not unpack:
            return extracted_data

        h_configs = []
        z_configs = []
        data_list = []

        for name, data in extracted_data.items():
            parts = name.split("_z")
            h_str = parts[0].replace("h", "")
            z_str = parts[1]

            h_configs.append(h_str)
            z_configs.append(int(z_str))
            data_list.append(data)

        return h_configs, z_configs, data_list

    except FileNotFoundError:
        print(f"File not found: {path}")
        return None