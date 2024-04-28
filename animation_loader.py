import streamlit as st
from streamlit_lottie import st_lottie
import json


def load_lottiefile(filepath: str):
    """ Load a Lottie animation from a JSON file located at filepath """
    with open(filepath, 'r') as file:
        return json.load(file)


def main():
    st.title('Lottie Animation Example')

    # Path to your Lottie JSON file
    lottie_animation_path = "Animation - 1714279612530.json"  # Replace this with your actual file path

    # Load your Lottie animation
    lottie_animation = load_lottiefile(lottie_animation_path)

    # Display Lottie animation in the Streamlit app
    st_lottie(lottie_animation, key="example")


if __name__ == "__main__":
    main()
