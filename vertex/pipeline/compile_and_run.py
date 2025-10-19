import argparse
import os
from kfp import dsl
from google_cloud_pipeline_components import aiplatform as gcc_aip


@dsl.pipeline(name="aster-basic-train-deploy")
def pipeline(project: str, region: str, model_dir: str, image_uri: str, bq_table: str):
    train_job = gcc_aip.CustomContainerTrainingJobRunOp(
        project=project,
        location=region,
        display_name="aster-train",
        container_uri=image_uri,
        args=[f"--BQ_TABLE={bq_table}", f"--AIP_MODEL_DIR={model_dir}"],
    )

    upload = gcc_aip.ModelUploadOp(
        project=project,
        location=region,
        display_name="aster-xgb",
        artifact_uri=model_dir,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest",
    )
    upload.set_caching_options(False)

    endpoint = gcc_aip.EndpointCreateOp(project=project, location=region, display_name="aster-endpoint")

    deploy = gcc_aip.ModelDeployOp(
        model=upload.outputs["model"],
        endpoint=endpoint.outputs["endpoint"],
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=2,
        dedicated_resources_machine_type="n1-standard-4",
    )
    deploy.after(upload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--bq-table", default="trading.features_daily")
    parser.add_argument("--image-uri", required=True)
    args = parser.parse_args()

    import google.cloud.aiplatform as aip
    from kfp import compiler

    aip.init(project=args.project, location=args.region, staging_bucket=args.pipeline_root)
    compiled = "pipeline.json"
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=compiled)

    model_dir = os.path.join(args.pipeline_root, "models/aster-sample")
    job = aip.PipelineJob(
        display_name="aster-basic-train-deploy",
        template_path=compiled,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "project": args.project,
            "region": args.region,
            "model_dir": model_dir,
            "image_uri": args.image_uri,
            "bq_table": args.bq_table,
        },
    )
    job.run(sync=False)


if __name__ == "__main__":
    main()


