from src.model import load_model
from src.config import MODEL_PATH

def main():
    # Initialize and load the model
    model = load_model(MODEL_PATH)
    print("ULLM successfully loaded. Ready for interaction.")

    # Interactive loop
    while True:
        try:
            user_input = input("Enter your query (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting ULLM. Goodbye!")
                break
            response = model.generate_response(user_input)
            print(f"Response: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
