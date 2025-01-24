"""MLCube handler file"""
import os 
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'  # see issue #152
import typer
import os

TASK_NAME = "BraTS-PED"
if TASK_NAME == "BraTS-SSA":
    import runner_ssa as runner
elif TASK_NAME == "BraTS-PED":
    import runner_ped as runner

app = typer.Typer()


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    output_path: str = typer.Option(..., "--output_path"),
):

    #runner.setup_model_weights()
    runner.batch_processor(data_path, output_path)
    return output_path

@app.command("install")
def install():
    runner.setup_model_weights()

@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()