# Voyagers Forecasting

This project is a readaptation (fork) of the [Chronos](https://github.com/amazon-science/chronos-forecasting) repository, designed for advanced time series forecasting experiments and development.

## Project Structure

The codebase is organized to separate legacy components from the core Chronos-2 implementation and future adaptations:

- **`src/legacy/`**: Contains utility functions and older Chronos modules.
- **`src/chronos2/`**: Contains all pre-existent Chronos-2 modules.

<<<<<<< HEAD
## 🚀 News
- **20 Oct 2025**: 🚀 [Chronos-2](https://huggingface.co/amazon/chronos-2) released. It offers _zero-shot_ support for univariate, multivariate, and covariate-informed forecasting tasks. Chronos-2 achieves the best performance on fev-bench, GIFT-Eval and Chronos Benchmark II amongst pretrained models. Check out [this notebook](notebooks/chronos-2-quickstart.ipynb) to get started with Chronos-2.
- **14 Feb 2025**: 🚀 Chronos-Bolt is now available on Amazon SageMaker JumpStart! Check out the [tutorial notebook](notebooks/deploy-chronos-to-amazon-sagemaker.ipynb) to learn how to deploy Chronos endpoints for production use in 3 lines of code.
- **12 Dec 2024**: 📊 We released [`fev`](https://github.com/autogluon/fev), a lightweight package for benchmarking time series forecasting models based on the [Hugging Face `datasets`](https://huggingface.co/docs/datasets/en/index) library.
- **26 Nov 2024**: ⚡️ Chronos-Bolt models released [on HuggingFace](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444). Chronos-Bolt models are more accurate (5% lower error), up to 250x faster and 20x more memory efficient than the original Chronos models of the same size!
- **13 Mar 2024**: 🚀 Chronos [paper](https://arxiv.org/abs/2403.07815) and inference code released.
=======
## Notebooks
>>>>>>> 76322e8854eebc1a4fc9f159bcbaa25fa0a10ffa

The `notebooks/` directory contains the primary workflows for data preparation and model training:

- **`00_data_setup.ipynb`**: Generates the necessary synthetic datasets.
- **`01_training_baseline.ipynb`**: Handles the model baseline training and benchmarking process.

## Resources

- **Hugging Face Team**: [voyagersnlppolito](https://huggingface.co/voyagersnlppolito)
- **Weights & Biases Team**: [voyagers-polito](https://wandb.ai/voyagers-polito/huggingface)
