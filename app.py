import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="PhD Portfolio | Machine Learning & HDC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(45deg, #8e2de2, #4a00e0, #3f51b5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0 !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        margin-top: 0 !important;
        font-weight: 400 !important;
    }
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #3f51b5;
        margin-top: 2rem !important;
    }
    .card {
        color: #3f51b5;
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #3f51b5;
        margin: 1rem 0;
    }
    .highlight {
        color: #3f51b5;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #3f51b5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #283593;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Navigation")
    
    nav = st.radio('Navigate', ["Home", "Research (WIP)", "Projects (WIP)", "Publications (WIP)", "Contact (WIP)"], label_visibility='hidden')
    
    st.markdown("---")
    st.markdown("### Connect")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.page_link('https://www.linkedin.com/in/richard-ky/', label='LinkedIn')

if nav == "Home":
    cols = st.columns([1, 3, 1])
    with cols[1]:
        fig = go.Figure()
        
        layers = [4, 7, 5, 3]
        colors = ["#8e2de2", "#4a00e0", "#3f51b5"]
        node_x = []
        node_y = []
        node_colors = []
        edge_x = []
        edge_y = []
        
        for i, layer_size in enumerate(layers):
            for j in range(layer_size):
                x = i
                y = j - layer_size/2
                node_x.append(x)
                node_y.append(y)
                node_colors.append(colors[i % len(colors)])
                
                if i < len(layers) - 1:
                    for k in range(layers[i+1]):
                        edge_x.extend([x, i+1, None])
                        edge_y.extend([y, k - layers[i+1]/2, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color="#c2c2c2"),
            mode="lines",
            hoverinfo="none"
        ))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(
                size=15,
                color=node_colors,
                line=dict(width=1, color="#ffffff"),
            ),
            hoverinfo="none"
        ))
        
        fig.update_layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h1 class='main-header'>Richard Ky</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>PhD Student in Computer Science</p>", unsafe_allow_html=True)
    
    st.markdown("### Exploring the intersection of Machine Learning, Hyperdimensional Computing, and Data Science")

    st.markdown("<h2 class='section-header'>About Me</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
    I'm a Computer Science PhD candidate specializing in <span class='highlight'>Machine Learning</span> and <span class='highlight'>Hyperdimensional Computing</span>.
    My research explores novel computational paradigms inspired by the human brain to develop more efficient and robust AI systems.

    Currently working on advancing the state-of-the-art in <span class='highlight'>neuro-symbolic AI</span> and <span class='highlight'>interpretable machine learning</span> 
    with applications in healthcare, autonomous systems, and scientific discovery.

    I'm passionate about creating AI systems that are not only powerful but also transparent, fair, and beneficial to society.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Holographic Reduced Representations Vector Space</h3>", unsafe_allow_html=True)
    
    @st.cache_data
    def generate_hrr_vectors(n=8, dim=100):
        phases = np.random.uniform(0, 2*np.pi, size=(n, dim))
        return np.exp(1j * phases)
    
    @st.cache_data
    def hrr_operations():
        vector_a = generate_hrr_vectors(1, 100)[0]
        vector_b = generate_hrr_vectors(1, 100)[0]
        
        binding_result = vector_a * vector_b
        
        superposition = (vector_a + vector_b) / 2
        
        return {
            "A": vector_a,
            "B": vector_b,
            "A⊗B": binding_result,
            "A+B": superposition
        }
    
    hrr_vectors = hrr_operations()
    
    tabs = st.tabs(["Unit Circle", "Phase Distribution", "Operations"])
    
    with tabs[0]:
        fig = go.Figure()
        
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='lightgray', width=1),
            hoverinfo='skip',
            name='Unit Circle'
        ))
        
        sample_size = 20
        for name, vector in list(hrr_vectors.items())[:2]:
            real = np.real(vector[:sample_size])
            imag = np.imag(vector[:sample_size])
            
            fig.add_trace(go.Scatter(
                x=real,
                y=imag,
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name=f'Vector {name}'
            ))
            
            for i in range(sample_size):
                fig.add_trace(go.Scatter(
                    x=[0, real[i]],
                    y=[0, imag[i]],
                    mode='lines',
                    line=dict(width=0.5),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        fig.update_layout(
            title="HRR Vectors on Complex Unit Circle (first 20 dimensions)",
            xaxis_title="Real",
            yaxis_title="Imaginary",
            xaxis=dict(range=[-1.2, 1.2], zeroline=True),
            yaxis=dict(range=[-1.2, 1.2], zeroline=True),
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        fig = go.Figure()
        
        for name, vector in hrr_vectors.items():
            phases = np.angle(vector)
            
            fig.add_trace(go.Histogram(
                x=phases,
                opacity=0.7,
                name=name,
                nbinsx=20
            ))
        
        fig.update_layout(
            title="Phase Angle Distribution",
            xaxis_title="Phase Angle (radians)",
            yaxis_title="Count",
            barmode='overlay',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.write("Explore HRR operations:")
        
        operation = st.selectbox(
            "Select operation",
            ["Binding (A⊗B)", "Superposition (A+B)", "Compare similarity"]
        )
        
        if operation == "Binding (A⊗B)":
            phases_a = np.angle(hrr_vectors["A"][:20])
            phases_b = np.angle(hrr_vectors["B"][:20])
            phases_bind = np.angle(hrr_vectors["A⊗B"][:20])
            
            df = pd.DataFrame({
                "Dimension": range(1, 21),
                "Vector A (phase)": phases_a,
                "Vector B (phase)": phases_b,
                "A⊗B (phase)": phases_bind
            })
            
            st.write("Phase angles after binding operation:")
            st.dataframe(df.style.format({"Vector A (phase)": "{:.2f}", 
                                            "Vector B (phase)": "{:.2f}", 
                                            "A⊗B (phase)": "{:.2f}"}))
            
            st.info("In HRR, binding (circular convolution) combines vectors to represent associations or relationships. In the frequency domain, this is simply element-wise multiplication of complex numbers.")
            
        elif operation == "Superposition (A+B)":
            fig = go.Figure()
            
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode='lines',
                line=dict(color='lightgray', width=1),
                hoverinfo='skip',
                showlegend=False
            ))
            
            super_vector = hrr_vectors["A+B"]
            real = np.real(super_vector[:20])
            imag = np.imag(super_vector[:20])
            
            fig.add_trace(go.Scatter(
                x=real,
                y=imag,
                mode='markers',
                marker=dict(
                    size=10,
                    color='purple',
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name='Superposition'
            ))
            
            fig.update_layout(
                title="Superposition Result (normalized)",
                xaxis_title="Real",
                yaxis_title="Imaginary",
                xaxis=dict(range=[-1.2, 1.2], zeroline=True),
                yaxis=dict(range=[-1.2, 1.2], zeroline=True),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Superposition in HRR allows multiple vectors to be combined while preserving their individual information. This operation is used to represent sets or collections.")
            
        else:
            def hrr_similarity(vec1, vec2):
                return np.abs(np.sum(np.conjugate(vec1) * vec2)) / (np.sqrt(np.sum(np.abs(vec1)**2)) * np.sqrt(np.sum(np.abs(vec2)**2)))
            
            similarities = {
                "A and B": hrr_similarity(hrr_vectors["A"], hrr_vectors["B"]),
                "A and A⊗B": hrr_similarity(hrr_vectors["A"], hrr_vectors["A⊗B"]),
                "B and A⊗B": hrr_similarity(hrr_vectors["B"], hrr_vectors["A⊗B"]),
                "A and A+B": hrr_similarity(hrr_vectors["A"], hrr_vectors["A+B"]),
                "B and A+B": hrr_similarity(hrr_vectors["B"], hrr_vectors["A+B"])
            }
            
            fig = go.Figure(go.Bar(
                x=list(similarities.keys()),
                y=list(similarities.values()),
                text=[f"{v:.3f}" for v in similarities.values()],
                textposition='auto',
                marker_color='royalblue'
            ))
            
            fig.update_layout(
                title="Vector Similarities",
                yaxis_title="Cosine Similarity",
                yaxis=dict(range=[0, 1]),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("This shows how similar different vectors are to each other. Note how binding creates a vector dissimilar to both inputs, while superposition creates a vector that has moderate similarity to its components.")

    st.markdown("<h2 class='section-header'>Research Interests</h2>", unsafe_allow_html=True)
    
    interests = {
        "Hyperdimensional Computing": "Developing brain-inspired computing models using high-dimensional vector spaces for efficient information representation and processing.",
        "Explainable AI": "Creating transparent machine learning models that can explain their decision-making processes to humans.",
        "Neuro-Symbolic Integration": "Combining neural networks with symbolic reasoning for more robust and interpretable AI systems.",
        "Data Visualization": "Designing interactive visual tools to gain insights from complex datasets and communicate findings effectively."
    }
    
    cols = st.columns(2)
    for i, (interest, description) in enumerate(interests.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div class='card'>
            <h3>{interest}</h3>
            <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-header'>Independent Research Projects</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h3>Computer Vision + HDC</h3>
        <p>Using HDC to encode scale-invariant feature transform (SIFT)-derived features for use in neural nets.</p>
        <br>
        <a href="#" target="_blank">Under Construction</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h3>NLP + HDC</h3>
        <p>Training Hidden Markov models on superimposed HDC bag-of-words encodings to create lightweight LLMs.</p>
        <br>
        <a href="#" target="_blank">Under Construction</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
        <h3>Graphs + HDC</h3>
        <p>Approximating graph neural networks and the Weisfeiler-Lehman kernel through HDC node embeddings.</p>
        <br>
        <a href="#" target="_blank">Under Construction</a>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
    with cta_col2:
        st.markdown("<h3 style='text-align: center;'>Interested in collaboration?</h3>", unsafe_allow_html=True)
        if st.button("Contact Me", use_container_width=True):
            st.success("Thanks for your interest! Please email me at kyr[at]uci.edu.")

elif nav == "Research":
    st.title("Research")
    st.info("This section is under development. Please check back soon!")
    
elif nav == "Projects":
    st.title("Projects")
    st.info("This section is under development. Please check back soon!")
    
elif nav == "Publications":
    st.title("Publications")
    st.info("This section is under development. Please check back soon!")
    
elif nav == "Contact":
    st.title("Contact")
    st.info("This section is under development. Please check back soon!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 0.8rem;'>© 2025 | Built with Streamlit</p>", unsafe_allow_html=True)