
    # Helper functions for working with nested DataFrame structure

    import pandas as pd
    import numpy as np

    def expand_predictions_df(df):
        """
        Expand nested predictions DataFrame to long format.

        Args:
            df: DataFrame with 'predictions' column containing lists

        Returns:
            DataFrame in long format with one row per (theta, model, position)
        """
        expanded_records = []
        for _, row in df.iterrows():
            for position, prob in enumerate(row['predictions'], 1):
                expanded_records.append({
                    'theta': row['theta'],
                    'model_name': row['model_name'],
                    'position': position,
                    'predicted_probability': prob,
                    'd_model': row['d_model'],
                    'd_head': row['d_head'],
                    'n_heads': row['n_heads'],
                    'd_mlp': row['d_mlp'],
                    'n_layers': row['n_layers'],
                    'log_loss_ratio': row['log_loss_ratio']
                })
        return pd.DataFrame(expanded_records)

    def expand_theoretical_df(df):
        """
        Expand nested theoretical DataFrame to long format.

        Args:
            df: DataFrame with 'theoretical_predictions_avg' column containing lists

        Returns:
            DataFrame in long format with one row per (theta, position)
        """
        expanded_records = []
        for _, row in df.iterrows():
            for position, prob in enumerate(row['theoretical_predictions_avg'], 1):
                expanded_records.append({
                    'theta': row['theta'],
                    'position': position,
                    'theoretical_probability': prob
                })
        return pd.DataFrame(expanded_records)

    def expand_sequences_df(df):
        """
        Expand nested sequences DataFrame to long format.

        Args:
            df: DataFrame with 'sequences' column containing lists of dicts

        Returns:
            DataFrame in long format with one row per sequence
        """
        expanded_records = []
        for _, row in df.iterrows():
            for seq_dict in row['sequences']:
                expanded_records.append({
                    'theta': row['theta'],
                    'sequence_idx': seq_dict['sequence_idx'],
                    'sequence_length': seq_dict['sequence_length'],
                    'sequence': seq_dict['sequence'],
                    'num_ones': seq_dict['num_ones'],
                    'num_zeros': seq_dict['num_zeros'],
                    'empirical_probability': seq_dict['empirical_probability']
                })
        return pd.DataFrame(expanded_records)

    # Example usage:
    # predictions_df = pd.read_csv('dimension_bottleneck_predictions.csv')
    # predictions_long = expand_predictions_df(predictions_df)
    