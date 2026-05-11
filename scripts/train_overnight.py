#!/usr/bin/env python3
"""
Overnight Training Script - Train Both CNN-LSTM and TCN

Trains both models sequentially on the same dataset split.
Safe to run unattended - comprehensive error handling and logging.

Outputs:
- models/cnn_lstm_overnight_TIMESTAMP/
- models/tcn_overnight_TIMESTAMP/
- training_overnight.log
- comparison_report.txt

Usage:
    python scripts/train_overnight.py --db "/path/to/database.db"

    # Or run in background (survives terminal close):
    nohup python scripts/train_overnight.py --db "/path/to/database.db" > training_overnight.log 2>&1 &
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger

from src.models.dataset import DatasetBuilder, create_dataloaders
from src.models.trainer import ModelTrainer
from src.models.metrics import ModelEvaluator
from src.models.validation import TimeSeriesSplitter


async def train_model(
    model_type: str,
    db_path: str,
    train_dataset,
    val_dataset,
    test_dataset,
    loaders: dict,
    timestamp: str
) -> dict:
    """
    Train a single model and return results.

    Returns:
        dict with 'success', 'model_dir', 'evaluation', 'error'
    """
    try:
        logger.info("=" * 80)
        logger.info(f"TRAINING {model_type.upper()} MODEL")
        logger.info("=" * 80)
        logger.info("")

        # Create checkpoint directory
        model_name = f"{model_type}_overnight_{timestamp}"
        checkpoint_dir = Path("models") / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trainer
        trainer = ModelTrainer(
            model_type=model_type,
            n_features=train_dataset.n_features,
            sequence_length=train_dataset.sequence_length,
            device='mps',
            loss_type='focal',
            focal_alpha=0.05,
            focal_gamma=4.0,
            learning_rate=0.001,
            weight_decay=1e-5,
            checkpoint_dir=str(checkpoint_dir),
            use_undersampling=True,
            undersampling_ratio=0.20
        )

        # Train
        logger.info("Starting training...")
        start_time = datetime.now()

        history = trainer.train(
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            epochs=50,
            early_stopping_patience=10,
            verbose=True
        )

        train_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
        logger.info("")

        trainer.save_history()

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        trainer.load_checkpoint('best_model.pt')
        trainer.model.eval()

        all_predictions = []
        all_labels = []
        all_probas = []

        with torch.no_grad():
            for sequences, labels in loaders['test']:
                sequences = sequences.to(trainer.device)
                outputs = trainer.model(sequences)

                predictions = (outputs > 0.5).float().cpu().numpy()
                probas = outputs.cpu().numpy()

                all_predictions.extend(predictions.flatten())
                all_labels.extend(labels.numpy().flatten())
                all_probas.extend(probas.flatten())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probas = np.array(all_probas)

        evaluator = ModelEvaluator()
        evaluation = evaluator.evaluate_model(
            y_true=all_labels,
            y_pred=all_predictions,
            y_proba=all_probas
        )

        evaluator.print_report(evaluation)
        evaluator.save_report(evaluation, str(checkpoint_dir / 'evaluation.json'))

        logger.info("")
        logger.success(f"✓ {model_type.upper()} training complete!")
        logger.info(f"  Model saved to: {checkpoint_dir}")
        logger.info(f"  Training time: {train_time:.1f}s")
        logger.info(f"  Test Accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"  Test Precision: {evaluation['precision']:.4f}")
        logger.info(f"  Test Recall: {evaluation['recall']:.4f}")
        logger.info(f"  Test F1: {evaluation['f1']:.4f}")
        logger.info("")

        return {
            'success': True,
            'model_dir': str(checkpoint_dir),
            'evaluation': evaluation,
            'train_time': train_time,
            'history': history,
            'error': None
        }

    except Exception as e:
        logger.error(f"✗ {model_type.upper()} training failed: {e}")
        logger.error(traceback.format_exc())

        return {
            'success': False,
            'model_dir': None,
            'evaluation': None,
            'train_time': None,
            'history': None,
            'error': str(e)
        }


async def main(db_path: str):
    """
    Main overnight training function.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 80)
    logger.info("OVERNIGHT TRAINING: CNN-LSTM + TCN")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Device: MPS (Apple Silicon)")
    logger.info("=" * 80)
    logger.info("")

    # ========================================================================
    # STEP 1: Build Dataset (shared by both models)
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: BUILDING DATASET")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Dataset parameters
        sequence_days = 3
        candles_per_day = 24 * 60  # 1-minute candles
        sequence_candles = int(sequence_days * candles_per_day)

        logger.info(f"Sequence: {sequence_candles} candles ({sequence_days} days @ 1m)")
        logger.info("")

        # Initialize dataset builder
        # Note: Pre-spike target generation is built into SpikeTargetGenerator
        # No need for forward_window or min_spike_threshold parameters
        builder = DatasetBuilder(
            db_path=db_path,
            sequence_length=sequence_candles,
            scaler_type='robust'
        )

        # Build dataset for all symbols
        logger.info("Loading data for all symbols...")
        dataset = builder.build_dataset_multi_symbol(
            symbols=None,  # All symbols
            timeframe='1m',
            start_date=None,  # All available data
            end_date=None,
            feature_columns=None,
            min_candles=sequence_candles + 100,
            max_symbols=None,  # No limit
            skip_symbols=0,
            normalize=False,
            fit_scaler=False
        )

        if dataset is None:
            logger.error("Failed to build dataset!")
            return

        logger.success(f"✓ Dataset created: {len(dataset)} samples")
        stats = dataset.get_statistics()
        logger.info(f"  Positive: {stats['positive_samples']} ({stats['positive_ratio']*100:.2f}%)")
        logger.info(f"  Negative: {stats['negative_samples']}")
        logger.info(f"  Features: {stats['n_features']}")
        logger.info("")

    except Exception as e:
        logger.error(f"Dataset building failed: {e}")
        logger.error(traceback.format_exc())
        return

    # ========================================================================
    # STEP 2: Split Dataset (60/20/20)
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 2: SPLITTING DATASET")
    logger.info("=" * 80)
    logger.info("")

    try:
        splitter = TimeSeriesSplitter(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            gap=2
        )

        train_idx, val_idx, test_idx = splitter.split(len(dataset))

        train_dataset = type(dataset)(
            sequences=dataset.sequences[train_idx].numpy(),
            labels=dataset.labels[train_idx].numpy(),
            timestamps=dataset.timestamps[train_idx],
            symbols=dataset.symbols[train_idx],
            feature_names=dataset.feature_names
        )

        val_dataset = type(dataset)(
            sequences=dataset.sequences[val_idx].numpy(),
            labels=dataset.labels[val_idx].numpy(),
            timestamps=dataset.timestamps[val_idx],
            symbols=dataset.symbols[val_idx],
            feature_names=dataset.feature_names
        )

        test_dataset = type(dataset)(
            sequences=dataset.sequences[test_idx].numpy(),
            labels=dataset.labels[test_idx].numpy(),
            timestamps=dataset.timestamps[test_idx],
            symbols=dataset.symbols[test_idx],
            feature_names=dataset.feature_names
        )

        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Val:   {len(val_dataset)} samples")
        logger.info(f"  Test:  {len(test_dataset)} samples")
        logger.info("")

    except Exception as e:
        logger.error(f"Dataset splitting failed: {e}")
        logger.error(traceback.format_exc())
        return

    # ========================================================================
    # STEP 3: Normalize
    # ========================================================================
    logger.info("Normalizing sequences...")

    try:
        train_sequences_norm = builder.normalize_sequences(train_dataset.sequences.numpy(), fit=True)
        val_sequences_norm = builder.normalize_sequences(val_dataset.sequences.numpy())
        test_sequences_norm = builder.normalize_sequences(test_dataset.sequences.numpy())

        train_dataset.sequences = torch.FloatTensor(train_sequences_norm)
        val_dataset.sequences = torch.FloatTensor(val_sequences_norm)
        test_dataset.sequences = torch.FloatTensor(test_sequences_norm)

        logger.success("✓ Normalization complete")
        logger.info("")

    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        logger.error(traceback.format_exc())
        return

    # ========================================================================
    # STEP 4: Undersample Training Set (1:5 ratio)
    # ========================================================================
    logger.info("Applying strategic undersampling (1:5 ratio)...")

    try:
        train_stats = train_dataset.get_statistics()
        n_positives = train_stats['positive_samples']
        n_negatives = train_stats['negative_samples']

        target_negatives = n_positives * 5
        keep_ratio = min(1.0, target_negatives / n_negatives)

        if keep_ratio < 1.0:
            positive_mask = train_dataset.labels == 1
            negative_mask = train_dataset.labels == 0

            positive_indices = np.where(positive_mask)[0]
            negative_indices = np.where(negative_mask)[0]

            np.random.seed(42)
            sampled_negative_indices = np.random.choice(
                negative_indices,
                size=int(len(negative_indices) * keep_ratio),
                replace=False
            )

            final_indices = np.concatenate([positive_indices, sampled_negative_indices])
            np.random.shuffle(final_indices)

            train_dataset = type(train_dataset)(
                sequences=train_dataset.sequences[final_indices].numpy(),
                labels=train_dataset.labels[final_indices].numpy(),
                timestamps=train_dataset.timestamps[final_indices],
                symbols=train_dataset.symbols[final_indices],
                feature_names=train_dataset.feature_names
            )

            logger.success(
                f"✓ Undersampled: {len(train_dataset)} samples "
                f"({n_positives} pos + {len(sampled_negative_indices)} neg)"
            )
        else:
            logger.info("No undersampling needed")

        logger.info("")

    except Exception as e:
        logger.error(f"Undersampling failed: {e}")
        logger.error(traceback.format_exc())
        return

    # ========================================================================
    # STEP 5: Create DataLoaders
    # ========================================================================
    logger.info("Creating data loaders...")

    try:
        loaders = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=16,
            use_weighted_sampler=True,
            num_workers=0
        )

        logger.success("✓ DataLoaders ready")
        logger.info("")

    except Exception as e:
        logger.error(f"DataLoader creation failed: {e}")
        logger.error(traceback.format_exc())
        return

    # ========================================================================
    # STEP 6: Train CNN-LSTM
    # ========================================================================
    cnn_lstm_result = await train_model(
        model_type='cnn_lstm',
        db_path=db_path,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        loaders=loaders,
        timestamp=timestamp
    )

    # ========================================================================
    # STEP 7: Train TCN
    # ========================================================================
    tcn_result = await train_model(
        model_type='tcn',
        db_path=db_path,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        loaders=loaders,
        timestamp=timestamp
    )

    # ========================================================================
    # STEP 8: Generate Comparison Report
    # ========================================================================
    logger.info("=" * 80)
    logger.info("GENERATING COMPARISON REPORT")
    logger.info("=" * 80)
    logger.info("")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("OVERNIGHT TRAINING RESULTS")
    report_lines.append("=" * 80)
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Database: {db_path}")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("DATASET STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total samples: {len(dataset)}")
    report_lines.append(f"Train samples: {len(train_dataset)}")
    report_lines.append(f"Val samples: {len(val_dataset)}")
    report_lines.append(f"Test samples: {len(test_dataset)}")
    report_lines.append(f"Features: {train_dataset.n_features}")
    report_lines.append(f"Sequence length: {train_dataset.sequence_length}")
    report_lines.append("")

    # CNN-LSTM results
    report_lines.append("=" * 80)
    report_lines.append("CNN-LSTM RESULTS")
    report_lines.append("=" * 80)
    if cnn_lstm_result['success']:
        eval_cnn = cnn_lstm_result['evaluation']
        report_lines.append(f"Status: ✓ SUCCESS")
        report_lines.append(f"Training time: {cnn_lstm_result['train_time']:.1f}s ({cnn_lstm_result['train_time']/60:.1f} min)")
        report_lines.append(f"Model directory: {cnn_lstm_result['model_dir']}")
        report_lines.append("")
        report_lines.append("Test Set Metrics:")
        report_lines.append(f"  Accuracy:  {eval_cnn['accuracy']:.4f}")
        report_lines.append(f"  Precision: {eval_cnn['precision']:.4f}")
        report_lines.append(f"  Recall:    {eval_cnn['recall']:.4f}")
        report_lines.append(f"  F1 Score:  {eval_cnn['f1']:.4f}")
        report_lines.append(f"  AUC-ROC:   {eval_cnn.get('auc_roc', 0):.4f}")
    else:
        report_lines.append(f"Status: ✗ FAILED")
        report_lines.append(f"Error: {cnn_lstm_result['error']}")
    report_lines.append("")

    # TCN results
    report_lines.append("=" * 80)
    report_lines.append("TCN RESULTS")
    report_lines.append("=" * 80)
    if tcn_result['success']:
        eval_tcn = tcn_result['evaluation']
        report_lines.append(f"Status: ✓ SUCCESS")
        report_lines.append(f"Training time: {tcn_result['train_time']:.1f}s ({tcn_result['train_time']/60:.1f} min)")
        report_lines.append(f"Model directory: {tcn_result['model_dir']}")
        report_lines.append("")
        report_lines.append("Test Set Metrics:")
        report_lines.append(f"  Accuracy:  {eval_tcn['accuracy']:.4f}")
        report_lines.append(f"  Precision: {eval_tcn['precision']:.4f}")
        report_lines.append(f"  Recall:    {eval_tcn['recall']:.4f}")
        report_lines.append(f"  F1 Score:  {eval_tcn['f1']:.4f}")
        report_lines.append(f"  AUC-ROC:   {eval_tcn.get('auc_roc', 0):.4f}")
    else:
        report_lines.append(f"Status: ✗ FAILED")
        report_lines.append(f"Error: {tcn_result['error']}")
    report_lines.append("")

    # Comparison (if both succeeded)
    if cnn_lstm_result['success'] and tcn_result['success']:
        report_lines.append("=" * 80)
        report_lines.append("COMPARISON")
        report_lines.append("=" * 80)

        winner_f1 = "CNN-LSTM" if eval_cnn['f1'] > eval_tcn['f1'] else "TCN"
        winner_precision = "CNN-LSTM" if eval_cnn['precision'] > eval_tcn['precision'] else "TCN"
        winner_speed = "TCN" if tcn_result['train_time'] < cnn_lstm_result['train_time'] else "CNN-LSTM"

        report_lines.append(f"Best F1 Score: {winner_f1}")
        report_lines.append(f"Best Precision: {winner_precision}")
        report_lines.append(f"Faster Training: {winner_speed}")
        report_lines.append("")

        speedup = cnn_lstm_result['train_time'] / tcn_result['train_time']
        report_lines.append(f"TCN Training Speedup: {speedup:.2f}x")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Print and save report
    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = Path("comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.success(f"✓ Comparison report saved to: {report_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.success("OVERNIGHT TRAINING COMPLETE!")
    logger.info("=" * 80)


def cli_main():
    parser = argparse.ArgumentParser(description='Train CNN-LSTM and TCN overnight')
    parser.add_argument('--db', required=True, help='Path to SQLite database with features')

    args = parser.parse_args()

    asyncio.run(main(args.db))


if __name__ == "__main__":
    cli_main()
