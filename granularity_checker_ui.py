import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.set_page_config(page_title="User Story Granularity Checker", layout="wide")
st.title("🔬 User Story Granularity Checker AI")

# ---- PROMPT ----
GRANULARITY_AGENT_PROMPT = """
You are an Agile requirements analyst and user story coach.

Your job is to:
- Decide if the following user story is granular (i.e., focused, specific, and achievable within a single sprint by one team).
- If granular, reply only with "Yes" and a brief rationale.
- If not granular, reply with "No", then explain why not, and suggest how to split or rewrite the story into smaller, granular stories if possible.

User Story:
{user_story}
"""

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])

def check_granularity(user_story):
    prompt = PromptTemplate.from_template(GRANULARITY_AGENT_PROMPT)
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"user_story": user_story})
    return response

with st.expander("ℹ️ About Granularity", expanded=False):
    st.markdown("""
    - **Granular user stories** are focused, small, and can be delivered in a single sprint.
    - This app uses an LLM agent to check your story and offers splitting advice.
    """)

# -------- Main UI Layout --------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Paste User Story")
    user_story = st.text_area(
        "Enter a single user story for analysis:",
        height=200,
        placeholder="E.g. As a user, I want to reset my password via email so that I can regain access if I forget my password."
    )
    submit = st.button("🔍 Check Granularity", use_container_width=True)
    st.markdown("— or —")
    csv_upload = st.file_uploader("Batch check (CSV with 'user_story' column)", type=["csv"])

with col2:
    st.subheader("🔎 Granularity Analysis Output")
    if submit and user_story.strip():
        with st.spinner("Analyzing story granularity..."):
            result = check_granularity(user_story.strip())
        if result.lower().startswith("yes"):
            st.success(result)
        else:
            st.error(result)
    elif submit:
        st.warning("Please enter a user story to check.")

    # Batch CSV mode
    import pandas as pd
    if csv_upload:
        df = pd.read_csv(csv_upload)
        if 'user_story' not in df.columns:
            st.error("CSV must have a 'user_story' column.")
        else:
            with st.spinner("Analyzing batch of user stories..."):
                results = [check_granularity(us) for us in df['user_story'].fillna("")]
                df["granularity_result"] = results
            st.dataframe(df[["user_story", "granularity_result"]], use_container_width=True)
            csv_out = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", data=csv_out, file_name="granularity_results.csv")

st.markdown("---")
st.caption("Built with Streamlit · Powered by GPT-4o · [YourOrg]")
