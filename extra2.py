import streamlit as st
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

# Load the CLIP model and processor from the local path
model_path = "clip-vit-base-patch32"  # Adjust this path accordingly
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

# Function to get search results from Pixabay
def get_pixabay_results(query):
    api_key = "46481189-6da79b491097cfd086a1ba620"  # Replace with your actual Pixabay API key
    url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo"
    
    response = requests.get(url)

    # Debugging print statements
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response content: {response.text}")
        return []  # Return an empty list if the request failed

    try:
        data = response.json()
        return data.get('hits', [])
    except ValueError as e:
        print(f"JSON decode error: {e}")
        print(f"Response content: {response.text}")
        return []  # Return an empty list if JSON decoding fails

def main():
    st.title("Intelligent Image and Prompt Search")

    # Upload image and input prompt
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    user_prompt = st.text_input("Enter your search prompt:")

    # Add a submit button
    if st.button("Submit"):
        if uploaded_image and user_prompt:
            # Load and display the uploaded image
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process the image and prompt with CLIP
            inputs = processor(text=[user_prompt], images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            # Display similarity score
            st.write(f"Similarity score for the image and the prompt '{user_prompt}': {probs.item():.4f}")

            # Create a focused query based on the user prompt
            enhanced_prompt = f"{user_prompt.split()[2]} products"  # Simplified query focusing on a single object type

            # Perform Pixabay search
            search_results = get_pixabay_results(enhanced_prompt)

            # Display search results
            st.subheader("Top Search Results:")
            for result in search_results[:5]:  # Show top 5 results
                title = result.get('tags', 'No title')
                link = result.get('pageURL', '#')
                thumbnail = result.get('previewURL')

                # Display the result title and link
                st.markdown(f"[{title}]({link})")

                # Display the thumbnail image if available
                if thumbnail:
                    st.image(thumbnail, caption=title, use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/150", caption="Placeholder Image", use_column_width=True)

if __name__ == "__main__":
    main()
