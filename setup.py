from setuptools import setup, find_namespace_packages

setup(name='nnsegnext',
      packages=find_namespace_packages(include=["nnsegnext", "nnsegnext.*"]),
      install_requires=[
            "torch>=1.6.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.21",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel", 'tifffile'
      ],
      entry_points={
          'console_scripts': [
              'nnSegnext_convert_decathlon_task = nnsegnext.experiment_planning.nnSegnext_convert_decathlon_task:main',
              'nnSegnext_plan_and_preprocess = nnsegnext.experiment_planning.nnSegnext_plan_and_preprocess:main',
              'nnSegnext_train = nnsegnext.run.run_training:main',
              'nnSegnext_train_DP = nnsegnext.run.run_training_DP:main',
              'nnSegnext_train_DDP = nnsegnext.run.run_training_DDP:main',
              'nnSegnext_predict = nnsegnext.inference.predict_simple:main',
              'nnSegnext_ensemble = nnsegnext.inference.ensemble_predictions:main',
              'nnSegnext_find_best_configuration = nnsegnext.evaluation.model_selection.figure_out_what_to_submit:main',
              'nnSegnext_print_available_pretrained_models = nnsegnext.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'nnSegnext_print_pretrained_model_info = nnsegnext.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'nnSegnext_download_pretrained_model = nnsegnext.inference.pretrained_models.download_pretrained_model:download_by_name',
              'nnSegnext_download_pretrained_model_by_url = nnsegnext.inference.pretrained_models.download_pretrained_model:download_by_url',
              'nnSegnext_determine_postprocessing = nnsegnext.postprocessing.consolidate_postprocessing_simple:main',
              'nnSegnext_export_model_to_zip = nnsegnext.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'nnSegnext_install_pretrained_model_from_zip = nnsegnext.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'nnSegnext_change_trainer_class = nnsegnext.inference.change_trainer:main',
              'nnSegnext_evaluate_folder = nnsegnext.evaluation.evaluator:nnSegnext_evaluate_folder',
              'nnSegnext_plot_task_pngs = nnsegnext.utilities.overlay_plots:entry_point_generate_overlay',
          ],
      },
      
      )
