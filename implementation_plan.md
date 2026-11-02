# Goal Description
Add functionality for a user to input a topic they want notes for. We will compute the user's overall learning level based on their assessment scores (Beginner, Intermediate, Advanced) and generate AI-tailored notes for that topic.

## User Review Required
> [!IMPORTANT]
> - Do you have a preferred AI API to use (e.g., Google Gemini, OpenAI)? This plan assumes **Google Gemini (`google-generativeai`)**.
> - You will need to provide your API key via an environment variable `GEMINI_API_KEY` when running the app. Is this acceptable?
> - The application currently predicts levels per subject (Math, Science, etc.). I plan to determine an **overall level** by taking the most frequent level across the subjects to generate the notes. Does that sound good, or would you prefer a different logic?

## Proposed Changes

### Frontend Modifications
#### [MODIFY] [results.html](file:///c:/Users/USER/OneDrive/Documents/project/templates/results.html)
- **MathJax Integration**: Add the MathJax CDN script to the `<head>` for rendering LaTeX math formulas.

### Backend Modifications
#### [MODIFY] [app.py](file:///c:/Users/USER/OneDrive/Documents/project/app.py)
- **Prompt Refinement**: Update the `generate_notes` prompt to explicitly request wrapping all mathematical formulas in double dollar signs (`$$`) for MathJax compatibility.

## Verification Plan

### Automated Tests
- N/A - The application currently does not have an automated testing suite.

### Manual Verification
1. Export the `GEMINI_API_KEY` in the terminal.
2. Run `python app.py`.
3. Open `http://127.0.0.1:5000/assessment`.
4. Fill in the form including the new "Topic" field (e.g., "Photosynthesis" or "Fractions").
5. Submit the form and verify that the results page successfully displays both the individual subject scores and the AI-generated notes tailored to the overall assessment level.
