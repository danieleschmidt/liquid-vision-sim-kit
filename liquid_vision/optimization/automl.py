"""
AutoML and hyperparameter optimization for liquid neural networks.
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

from ..core.liquid_neurons import LiquidNet, create_liquid_net
from ..training.liquid_trainer import LiquidTrainer, TrainingConfig
from ..utils.logging import log_performance, performance_logger
from ..utils.validation import validate_inputs


logger = logging.getLogger('liquid_vision.optimization.automl')


@dataclass
class HyperparameterSpace:
    """Hyperparameter search space definition."""
    
    # Model architecture parameters
    hidden_units: List[List[int]]
    liquid_time_constant: List[float]
    liquid_dropout: List[float]
    
    # Training parameters
    learning_rate: List[float]
    batch_size: List[int]
    optimizer: List[str]
    weight_decay: List[float]
    
    # Data parameters
    encoder_type: List[str]
    time_window: List[float]
    
    def sample_random(self) -> Dict[str, Any]:
        """Sample random hyperparameters from space."""
        return {
            'hidden_units': random.choice(self.hidden_units),
            'liquid_time_constant': random.choice(self.liquid_time_constant),
            'liquid_dropout': random.choice(self.liquid_dropout),
            'learning_rate': random.choice(self.learning_rate),
            'batch_size': random.choice(self.batch_size),
            'optimizer': random.choice(self.optimizer),
            'weight_decay': random.choice(self.weight_decay),
            'encoder_type': random.choice(self.encoder_type),
            'time_window': random.choice(self.time_window),
        }
    
    def grid_search_space(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        keys = ['hidden_units', 'liquid_time_constant', 'liquid_dropout',
                'learning_rate', 'batch_size', 'optimizer', 'weight_decay', 
                'encoder_type', 'time_window']
        
        values = [getattr(self, key) for key in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations


class HPOptimizer:
    """
    Hyperparameter optimization using various strategies.
    """
    
    def __init__(
        self,
        objective_metric: str = "val_accuracy",
        optimization_direction: str = "maximize",  # maximize or minimize
        n_trials: int = 50,
        timeout_hours: float = 24.0,
        early_stopping_patience: int = 10,
        parallel_jobs: int = 1
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            objective_metric: Metric to optimize
            optimization_direction: Whether to maximize or minimize metric
            n_trials: Number of trials to run
            timeout_hours: Maximum optimization time
            early_stopping_patience: Early stopping patience
            parallel_jobs: Number of parallel optimization jobs
        """
        self.objective_metric = objective_metric
        self.optimization_direction = optimization_direction
        self.n_trials = n_trials
        self.timeout_hours = timeout_hours
        self.early_stopping_patience = early_stopping_patience
        self.parallel_jobs = parallel_jobs
        
        self.trial_results = []
        self.best_params = None
        self.best_score = float('-inf') if optimization_direction == "maximize" else float('inf')
        
    def random_search(
        self,
        hyperparameter_space: HyperparameterSpace,
        evaluation_function: Callable,
        base_config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Perform random search optimization.
        
        Args:
            hyperparameter_space: Space to search
            evaluation_function: Function to evaluate configurations
            base_config: Base training configuration
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting random search with {self.n_trials} trials")
        
        start_time = time.time()
        trials_without_improvement = 0
        
        for trial_idx in range(self.n_trials):
            # Check timeout
            if time.time() - start_time > self.timeout_hours * 3600:
                logger.warning("Timeout reached, stopping optimization")
                break
            
            # Sample hyperparameters
            params = hyperparameter_space.sample_random()
            
            # Evaluate configuration
            try:
                result = self._evaluate_configuration(
                    params, evaluation_function, base_config, trial_idx
                )
                
                self.trial_results.append(result)
                
                # Check if best
                if self._is_better(result['score']):
                    self.best_score = result['score']
                    self.best_params = params.copy()
                    trials_without_improvement = 0
                    logger.info(f"New best score: {self.best_score:.4f}")
                else:
                    trials_without_improvement += 1
                
                # Early stopping
                if trials_without_improvement >= self.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
                    
            except Exception as e:
                logger.error(f"Trial {trial_idx} failed: {e}")
                continue
        
        return self._prepare_results()
    
    def grid_search(
        self,
        hyperparameter_space: HyperparameterSpace,
        evaluation_function: Callable,
        base_config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            hyperparameter_space: Space to search
            evaluation_function: Function to evaluate configurations
            base_config: Base training configuration
            
        Returns:
            Optimization results
        """
        param_combinations = hyperparameter_space.grid_search_space()
        
        # Limit combinations if too many
        if len(param_combinations) > self.n_trials:
            logger.warning(f"Grid search has {len(param_combinations)} combinations, "
                          f"limiting to {self.n_trials}")
            param_combinations = random.sample(param_combinations, self.n_trials)
        
        logger.info(f"Starting grid search with {len(param_combinations)} combinations")
        
        if self.parallel_jobs > 1:
            return self._parallel_grid_search(
                param_combinations, evaluation_function, base_config
            )
        else:
            return self._sequential_grid_search(
                param_combinations, evaluation_function, base_config
            )
    
    def bayesian_optimization(
        self,
        hyperparameter_space: HyperparameterSpace,
        evaluation_function: Callable,
        base_config: TrainingConfig,
        n_initial: int = 5
    ) -> Dict[str, Any]:
        """
        Bayesian optimization using Gaussian Process surrogate model.
        This is a simplified implementation - in production use libraries like Optuna.
        
        Args:
            hyperparameter_space: Space to search
            evaluation_function: Function to evaluate configurations
            base_config: Base training configuration
            n_initial: Number of initial random trials
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting Bayesian optimization with {self.n_trials} trials")
        
        # Initial random exploration
        for trial_idx in range(min(n_initial, self.n_trials)):
            params = hyperparameter_space.sample_random()
            
            try:
                result = self._evaluate_configuration(
                    params, evaluation_function, base_config, trial_idx
                )
                
                self.trial_results.append(result)
                
                if self._is_better(result['score']):
                    self.best_score = result['score']
                    self.best_params = params.copy()
                    
            except Exception as e:
                logger.error(f"Initial trial {trial_idx} failed: {e}")
                continue
        
        # Bayesian optimization iterations
        for trial_idx in range(n_initial, self.n_trials):
            # In a full implementation, use GP to suggest next parameters
            # For now, use random search with bias towards best regions
            params = self._suggest_next_params(hyperparameter_space)
            
            try:
                result = self._evaluate_configuration(
                    params, evaluation_function, base_config, trial_idx
                )
                
                self.trial_results.append(result)
                
                if self._is_better(result['score']):
                    self.best_score = result['score']
                    self.best_params = params.copy()
                    logger.info(f"New best score: {self.best_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Bayesian trial {trial_idx} failed: {e}")
                continue
        
        return self._prepare_results()
    
    def _evaluate_configuration(
        self,
        params: Dict[str, Any],
        evaluation_function: Callable,
        base_config: TrainingConfig,
        trial_idx: int
    ) -> Dict[str, Any]:
        """Evaluate a hyperparameter configuration."""
        logger.info(f"Evaluating trial {trial_idx}: {params}")
        
        start_time = time.time()
        
        # Create modified config
        config = TrainingConfig(**{**asdict(base_config), **params})
        
        # Evaluate
        metrics = evaluation_function(config)
        
        eval_time = time.time() - start_time
        score = metrics.get(self.objective_metric, 0.0)
        
        result = {
            'trial_idx': trial_idx,
            'params': params,
            'score': score,
            'metrics': metrics,
            'eval_time': eval_time,
            'timestamp': time.time()
        }
        
        logger.info(f"Trial {trial_idx} completed: score={score:.4f}, time={eval_time:.1f}s")
        return result
    
    def _is_better(self, score: float) -> bool:
        """Check if score is better than current best."""
        if self.optimization_direction == "maximize":
            return score > self.best_score
        else:
            return score < self.best_score
    
    def _suggest_next_params(self, hyperparameter_space: HyperparameterSpace) -> Dict[str, Any]:
        """
        Suggest next parameters for Bayesian optimization.
        Simplified implementation - in production use proper acquisition functions.
        """
        if not self.trial_results:
            return hyperparameter_space.sample_random()
        
        # Sort trials by score
        sorted_trials = sorted(
            self.trial_results,
            key=lambda x: x['score'],
            reverse=(self.optimization_direction == "maximize")
        )
        
        # Use top 20% of trials to guide search
        top_trials = sorted_trials[:max(1, len(sorted_trials) // 5)]
        
        # Sample with bias towards good parameters
        if random.random() < 0.5 and top_trials:
            # Mutate a good configuration
            base_params = random.choice(top_trials)['params'].copy()
            
            # Randomly mutate some parameters
            for key in base_params:
                if random.random() < 0.3:  # 30% mutation rate
                    space_values = getattr(hyperparameter_space, key, [base_params[key]])
                    base_params[key] = random.choice(space_values)
            
            return base_params
        else:
            # Random exploration
            return hyperparameter_space.sample_random()
    
    def _parallel_grid_search(
        self,
        param_combinations: List[Dict[str, Any]],
        evaluation_function: Callable,
        base_config: TrainingConfig
    ) -> Dict[str, Any]:
        """Parallel grid search execution."""
        logger.info(f"Running parallel grid search with {self.parallel_jobs} workers")
        
        with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            # Submit all trials
            future_to_trial = {
                executor.submit(
                    self._evaluate_configuration,
                    params, evaluation_function, base_config, idx
                ): (idx, params)
                for idx, params in enumerate(param_combinations)
            }
            
            # Collect results
            for future in as_completed(future_to_trial):
                trial_idx, params = future_to_trial[future]
                
                try:
                    result = future.result()
                    self.trial_results.append(result)
                    
                    if self._is_better(result['score']):
                        self.best_score = result['score']
                        self.best_params = params.copy()
                        logger.info(f"New best score: {self.best_score:.4f}")
                        
                except Exception as e:
                    logger.error(f"Parallel trial {trial_idx} failed: {e}")
        
        return self._prepare_results()
    
    def _sequential_grid_search(
        self,
        param_combinations: List[Dict[str, Any]],
        evaluation_function: Callable,
        base_config: TrainingConfig
    ) -> Dict[str, Any]:
        """Sequential grid search execution."""
        start_time = time.time()
        
        for trial_idx, params in enumerate(param_combinations):
            # Check timeout
            if time.time() - start_time > self.timeout_hours * 3600:
                logger.warning("Timeout reached, stopping grid search")
                break
            
            try:
                result = self._evaluate_configuration(
                    params, evaluation_function, base_config, trial_idx
                )
                
                self.trial_results.append(result)
                
                if self._is_better(result['score']):
                    self.best_score = result['score']
                    self.best_params = params.copy()
                    logger.info(f"New best score: {self.best_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Grid search trial {trial_idx} failed: {e}")
                continue
        
        return self._prepare_results()
    
    def _prepare_results(self) -> Dict[str, Any]:
        """Prepare optimization results."""
        if not self.trial_results:
            return {"status": "failed", "message": "No successful trials"}
        
        # Sort trials by score
        sorted_trials = sorted(
            self.trial_results,
            key=lambda x: x['score'],
            reverse=(self.optimization_direction == "maximize")
        )
        
        results = {
            "status": "completed",
            "best_score": self.best_score,
            "best_params": self.best_params,
            "best_trial": sorted_trials[0],
            "total_trials": len(self.trial_results),
            "successful_trials": len([t for t in self.trial_results if t['score'] != 0]),
            "optimization_direction": self.optimization_direction,
            "objective_metric": self.objective_metric,
            "all_trials": sorted_trials,
            "statistics": self._compute_statistics()
        }
        
        return results
    
    def _compute_statistics(self) -> Dict[str, float]:
        """Compute optimization statistics."""
        if not self.trial_results:
            return {}
        
        scores = [t['score'] for t in self.trial_results]
        eval_times = [t['eval_time'] for t in self.trial_results]
        
        return {
            "score_mean": np.mean(scores),
            "score_std": np.std(scores),
            "score_median": np.median(scores),
            "eval_time_mean": np.mean(eval_times),
            "eval_time_total": np.sum(eval_times)
        }


class AutoMLOptimizer:
    """
    Complete AutoML pipeline for liquid neural networks.
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        optimization_budget_hours: float = 6.0,
        early_stopping_patience: int = 5
    ):
        """
        Initialize AutoML optimizer.
        
        Args:
            task_type: Type of task (classification, regression)
            optimization_budget_hours: Total optimization time budget
            early_stopping_patience: Early stopping patience
        """
        self.task_type = task_type
        self.optimization_budget_hours = optimization_budget_hours
        self.early_stopping_patience = early_stopping_patience
        
        self.optimization_results = {}
        
    @log_performance("automl_optimization")
    def optimize(
        self,
        train_dataset,
        val_dataset,
        input_dim: int,
        output_dim: int,
        base_config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """
        Run complete AutoML optimization.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset  
            input_dim: Input dimension
            output_dim: Output dimension
            base_config: Base configuration to start from
            
        Returns:
            AutoML results with best model and configuration
        """
        logger.info("Starting AutoML optimization pipeline")
        
        # Create base configuration if not provided
        if base_config is None:
            base_config = TrainingConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                epochs=50,  # Reduced for AutoML
                batch_size=32
            )
        
        # Define hyperparameter search space
        search_space = self._create_search_space(input_dim, output_dim)
        
        # Create evaluation function
        def evaluate_config(config: TrainingConfig) -> Dict[str, float]:
            return self._evaluate_model_config(
                config, train_dataset, val_dataset
            )
        
        # Run hyperparameter optimization
        hp_optimizer = HPOptimizer(
            objective_metric="val_accuracy" if self.task_type == "classification" else "val_loss",
            optimization_direction="maximize" if self.task_type == "classification" else "minimize",
            n_trials=30,  # Reasonable number for AutoML
            timeout_hours=self.optimization_budget_hours * 0.8,  # Reserve time for final training
            early_stopping_patience=self.early_stopping_patience,
            parallel_jobs=2
        )
        
        # Try different optimization strategies
        optimization_results = []
        
        # Random search
        logger.info("Phase 1: Random search optimization")
        random_results = hp_optimizer.random_search(
            search_space, evaluate_config, base_config
        )
        optimization_results.append(("random_search", random_results))
        
        # Bayesian optimization starting from best random results
        logger.info("Phase 2: Bayesian optimization")
        if random_results["status"] == "completed":
            hp_optimizer.best_params = random_results["best_params"]
            hp_optimizer.best_score = random_results["best_score"]
            hp_optimizer.trial_results = random_results["all_trials"]
        
        bayesian_results = hp_optimizer.bayesian_optimization(
            search_space, evaluate_config, base_config, n_initial=5
        )
        optimization_results.append(("bayesian_optimization", bayesian_results))
        
        # Select best overall results
        best_results = self._select_best_results(optimization_results)
        
        # Final training with best configuration
        logger.info("Phase 3: Final model training")
        best_config = TrainingConfig(**{**asdict(base_config), **best_results["best_params"]})
        best_config.epochs = base_config.epochs  # Use full epochs for final training
        
        final_model, final_metrics = self._train_final_model(
            best_config, train_dataset, val_dataset
        )
        
        # Prepare final results
        automl_results = {
            "status": "completed",
            "task_type": self.task_type,
            "optimization_budget_hours": self.optimization_budget_hours,
            "best_configuration": asdict(best_config),
            "best_model": final_model,
            "final_metrics": final_metrics,
            "optimization_history": optimization_results,
            "best_trial_details": best_results,
            "total_trials": sum(len(r[1].get("all_trials", [])) for r in optimization_results),
            "optimization_time": sum(r[1].get("statistics", {}).get("eval_time_total", 0) for r in optimization_results)
        }
        
        logger.info("AutoML optimization completed successfully")
        logger.info(f"Best {hp_optimizer.objective_metric}: {best_results['best_score']:.4f}")
        
        return automl_results
    
    def _create_search_space(self, input_dim: int, output_dim: int) -> HyperparameterSpace:
        """Create hyperparameter search space."""
        # Scale hidden units based on input/output dimensions
        base_hidden = max(64, min(256, input_dim // 2))
        
        return HyperparameterSpace(
            hidden_units=[
                [base_hidden],
                [base_hidden, base_hidden // 2],
                [base_hidden * 2, base_hidden],
                [base_hidden, base_hidden, base_hidden // 2],
            ],
            liquid_time_constant=[10.0, 20.0, 30.0, 50.0],
            liquid_dropout=[0.0, 0.05, 0.1, 0.15],
            learning_rate=[0.0001, 0.0005, 0.001, 0.003, 0.01],
            batch_size=[16, 32, 64, 128],
            optimizer=["adam", "adamw", "sgd"],
            weight_decay=[0.0, 1e-5, 1e-4, 1e-3],
            encoder_type=["temporal", "spatial", "voxel"],
            time_window=[25.0, 50.0, 100.0]
        )
    
    def _evaluate_model_config(
        self,
        config: TrainingConfig,
        train_dataset,
        val_dataset
    ) -> Dict[str, float]:
        """Evaluate a model configuration."""
        try:
            # Create model
            model = create_liquid_net(
                config.architecture,
                config.input_dim,
                config.output_dim,
                config.hidden_units
            )
            
            # Create trainer
            trainer = LiquidTrainer(model, config)
            
            # Train model
            results = trainer.train(train_dataset, val_dataset)
            
            # Return metrics
            return {
                "val_accuracy": results.get("val_accuracy", 0.0),
                "val_loss": results.get("val_loss", float('inf')),
                "train_accuracy": results.get("train_accuracy", 0.0),
                "train_loss": results.get("train_loss", float('inf')),
                "training_time": results.get("training_time", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {
                "val_accuracy": 0.0,
                "val_loss": float('inf'),
                "train_accuracy": 0.0,
                "train_loss": float('inf'),
                "training_time": 0.0
            }
    
    def _select_best_results(self, optimization_results: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Select best results from multiple optimization runs."""
        best_score = float('-inf')
        best_results = None
        
        for strategy, results in optimization_results:
            if results["status"] == "completed":
                score = results["best_score"]
                if score > best_score:
                    best_score = score
                    best_results = results
        
        if best_results is None:
            raise RuntimeError("No successful optimization runs")
        
        return best_results
    
    def _train_final_model(
        self,
        config: TrainingConfig,
        train_dataset,
        val_dataset
    ) -> Tuple[LiquidNet, Dict[str, float]]:
        """Train final model with best configuration."""
        # Create model
        model = create_liquid_net(
            config.architecture,
            config.input_dim,
            config.output_dim,
            config.hidden_units
        )
        
        # Create trainer
        trainer = LiquidTrainer(model, config)
        
        # Train model
        results = trainer.train(train_dataset, val_dataset)
        
        return model, results