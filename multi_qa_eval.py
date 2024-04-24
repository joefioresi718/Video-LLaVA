import json
import tqdm
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="print scores for each benchmark")
    parser.add_argument("--model_name", default="videollava-7b", type=str, help="custom model name")
    args = parser.parse_args()
    return args


def eval(results):
    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in tqdm(results.items()):
        try:
            # Computing score
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except:
            print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)
    print()
    print()


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    print(f'Model: {args.model_name}')
    print()
    path_to_folders = './eval/GPT_Zero_Shot_QA'
    all_benchmarks = ['Activitynet_Zero_Shot_QA', 'MSRVTT_Zero_Shot_QA', 'MSVD_Zero_Shot_QA', 'TGIF_Zero_Shot_QA']

    for benchmark in all_benchmarks:
        print(f'Dataset: {benchmark}')
        # try:
        path_to_json = os.path.join(path_to_folders, benchmark, args.model_name, 'results.json')
        print(path_to_json)
        with open(path_to_json, "r") as json_file:
            results = json.load(json_file)
        eval(results)
        # except:
        #     print(f'No results found for {benchmark}.')
        #     print()
        #     print()


if __name__ == '__main__':
    main()
