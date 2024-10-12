import streamlit as st
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

# Load the CLIP model and processor from the local path
model_path = "clip-vit-base-patch32"  # Adjust this path accordingly
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

# Function to modify the user's query for better search results
def enhance_query(user_prompt):
    # Add relevant keywords for image search
    enhanced_prompt = f"{user_prompt} photo image high-quality object"
    return enhanced_prompt

# Function to get search results from Pixabay
def get_pixabay_results(query):
    api_key = "46481189-6da79b491097cfd086a1ba620"  # Replace with your actual Pixabay API key
    url = f"https://pixabay.com/api/?key={api_key}&q={query}&image_type=photo"
    response = requests.get(url)
    data = response.json()
    return data.get('hits', [])

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

            # Modify the user query to enhance search results
            enhanced_prompt = enhance_query(user_prompt)
            st.write(f"Enhanced search query: '{enhanced_prompt}'")

            # Perform Pixabay search with the enhanced query
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
