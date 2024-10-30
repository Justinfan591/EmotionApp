import streamlit as st
from algorithm import Algorithm
import matplotlib.pyplot as plt
import io
import nltk



def main():

    st.title("Emotion Analysis of Novels")

    st.write("""
    This application analyzes the emotions present in a novel using a pre-trained RoBERTa model fine-tuned on the GoEmotions dataset.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a text file of the novel", type=["txt"])

    if uploaded_file is not None:
        # Read the text file
        text = uploaded_file.read().decode('utf-8')

        if st.button("Analyze"):
            # Initialize the progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize the algorithm
            algorithm = Algorithm()

            # Define a progress callback function
            def progress_callback(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing: {int(progress * 100)}%")

            with st.spinner("Analyzing..."):
                # Generate the plot
                fig = algorithm.plot_emotion_graph(text, progress_callback=progress_callback)
                st.pyplot(fig)

                # Provide option to download the figure
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                st.download_button(
                    label="Download graph as PNG",
                    data=buf,
                    file_name="emotion_graph.png",
                    mime="image/png"
                )

            # Complete the progress bar
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

if __name__ == "__main__":
    nltk.data.path.append('./nltk_data/')
    nltk.download('punkt', download_dir='./nltk_data/', quiet=True)
    main()

