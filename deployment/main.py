from traffic_capture import capture
from flow_generator import generate_csv
from predictor import predict

def run_pipeline():

    capture()

    generate_csv()

    result = predict()

    attacks = result["attacks"]

    if not attacks:

        print("No attacks detected")

        return

    print("\nAttacks Detected:\n")

    for attack in attacks:

        print(
            f"{attack['prediction']} "
            f"from {attack['src_ip']} "
            f"to {attack['dst_ip']}"
        )


if __name__ == "__main__":

    run_pipeline()