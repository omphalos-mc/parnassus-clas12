## CMS 2011 Experiment

This directory contains the implementation of the CMS 2011 experiment, which is part of the `parnassus-core` project.

The full generation chain consists of three stages:
1. Event-level model, which generates event-level quantities, such as $N_{part}, E_{x}^{miss}, E_{y}^{miss}, H_T$.
    The config for this model is located in `configs/event.yaml`.
2. Particle-level model, which generates particle-level quantities, such as $p_{T}, \eta, \phi, v_x, v_y, v_z$. It uses the event-level model outputs as a additional conditioning.
    The config for this model is located in `configs/particle.yaml`.
3. Impact-level model, which generates impact parameter values, such as $d_{0}, \sigma_{d_{0}}, z_{0}, \sigma_{z_{0}}$. It uses the particle-level model outputs as a additional conditioning.
    The config for this model is located in `configs/impact.yaml`.

### Training

To train the models, you can run the following command from the `src/parnassus_core/experiments/cms_2011/` directory:

```bash
python main.py fit -c configs/event.yaml # For event-level model
python main.py fit -c configs/particle.yaml # For particle-level model
python main.py fit -c configs/impact.yaml # For impact-level model
```
<details>
<summary>"Running a test training"</summary>
    To test the training script you can run use the `--trainer.fast_dev_run` flag which will run over a small number of training and validation batches and then exit.

    ```bash
    python main.py fit -c configs/event.yaml --trainer.fast_dev_run
    ```
</details>

### Evaluation

To evaluate the models, you can run the evaluation script from the `src/parnassus_core/experiments/cms_2011/evaluation/` directory:

```bash
python eval.py -c config.yaml
```

This will run the evaluation on the test datasets specified in the `config.yaml` file and save the results to the `evals/` directory.

To launch this kind of evaluation you need to have `event` and `particle` models trained. The `impact` model is optional for the evaluation and can be skipped (just comment out the `impact_model` section in the `config.yaml` file).


### Evaluation Postprocessing

To postprocess the evaluation results, you can run the following command from the `src/parnassus_core/experiments/cms_2011/evaluation/` directory:

```bash
python postprocess_eval.py \
    -i <path_to_eval_root_file> \
    -d <path_to_data_file> \
    -o <output_filename> \
    -n <number_of_events (default 100)> \
```

Data file should be the same as used in the evaluation config file, and the output filename will be used to save the postprocessed results.

#### Optional flags for postprocessing

You can also use the following optional flags:

- `--eta`: If you want to apply an eta cut, you can specify the value with this flag (default is 3.0).
- `-dr`: Specify the delta R for jet clustering (default is 0.5).
- `--ip`: If you want to use impact parameters in the postprocessing, you can add this flag (default `False`).
- `-m [--matching]`: If you want to perform particle matching, you can add this flag (default `False`).
- `--dr_cut`: Specify the delta R cut for particle matching (default is 0.6).
