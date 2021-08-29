#!/usr/bin/env python
"""
 Performs basic cleaning on the data and save the results in Weights & Biases 
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

#   Implement in the section marked # YOUR CODE HERE # the steps we have implemented in the notebook,
#   including downloading the data from W&B. Remember to use the logger instance already provided 
#   to print meaningful messages to screen. Make sure to use args.min_price and args.max_price when 
#   dropping the outliers (instead of hard-coding the values like we did in the notebook). 
  
#   Save the results to a CSV file called clean_sample.csv (df.to_csv("clean_sample.csv", index=False)). 
#   NOTE: Remember to use index=False when saving to CSV, otherwise the data checks in the next step might 
#   fail because there will be an extra index column.  


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################

    logger.info(f"Downloading artifact {args.input_artifact}")

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info(f"cleaning artifact with min & max prices {args.min_price, args.max_price}")
    
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info(f"Saving cleaned data to {args.output_artifact}")
    # save cleaned data
    df.to_csv(args.output_artifact, index=False)

    logger.info(f"Logging artifact {args.output_artifact}") 
    # log artifact to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)

    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="this steps cleans the data")

    # The new step should accept the parameters:
    #  - input_artifact (the input artifact), 
    #  - output_artifact (the name for the output artifact),
    #  - output_type (the type for the output artifact), 
    #  - output_description (a description for the output artifact), 
    #  - min_price (the minimum price to consider) 
    #  - max_price (the maximum price to consider):

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="input artifact - sample csv file",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="name for the output artifact (e.g cleaned data.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="type of the output (e.g clean_sample)",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="description of the output (e.g data with outliers and null values removed)",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="the minimum price to consider and filter the price data for (e.g 50)",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="the maximum price to consider and filter the price data for (e.g 150)",
        required=True
    )

    args = parser.parse_args()

    go(args)
