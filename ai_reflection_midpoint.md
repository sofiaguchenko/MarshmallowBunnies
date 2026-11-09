# GenAI Reflection — Midpoint

**Project:** Judging a Book by Its Cover — Genre Bias & Visual Stereotypes in AI  
**Team:** Marshmallow Bunnies  
**Course:** Visualization for AI

**AI-aided pair:**
- Milana Koval — k12148486
- Sofiia Guchenko — k12148479

## 1. How We Used Generative AI

During our project, we used ChatGPT as an assistant for planning experiments, writing Python code, debugging, creating visualizations, and interpreting results.

Our research focuses on how CLIP classifies book covers into different genres and whether visual stereotypes influence its predictions.

ChatGPT helped us turn general research questions into smaller experiments. For example, we investigated classification errors, brightness, the presence of faces, prompt sensitivity, and prediction confidence.

We also used ChatGPT to develop an interactive Streamlit dashboard for exploring our results.

## 2. What Worked Well

The biggest advantage of using ChatGPT was saving time during implementation and debugging.

It helped us write code for data processing and visualizations without having to build every component from scratch. This allowed us to spend more time exploring the model's behavior.

ChatGPT was also useful for suggesting follow-up analyses. Instead of stopping after calculating overall accuracy, we investigated why certain genres were confused and whether visual characteristics could explain some errors.

Another useful aspect was the interactive workflow. We could share errors or unexpected results, ask for explanations, and improve the code step by step.

## 3. Challenges and Limitations

Not every AI suggestion worked immediately.

Some code needed modifications to fit our notebook, existing variables, and file structure. For example, after moving the dashboard and prediction data into separate folders, we had to adjust the file paths.

Another challenge was interpreting the results correctly. ChatGPT could suggest possible explanations for classification patterns, but these explanations were not always supported by the data.

We also learned that more complex approaches are not necessarily better. For example, descriptive genre prompts reduced CLIP's classification accuracy compared with neutral prompts.

These experiences showed us that AI-generated suggestions should be treated as starting points rather than final solutions.

## 4. How We Verified AI Outputs

We did not rely only on ChatGPT's explanations.

We executed the generated code, inspected the visualizations, and compared the results with our research questions.

When something did not work, we provided the error messages or outputs and asked ChatGPT to help identify the problem.

We also tried to distinguish observations from explanations. For example, finding a relationship between cover brightness and genre predictions does not prove that brightness directly causes the model's decisions.

For confidence analysis, we considered the difference between CLIP's similarity-based scores and calibrated probabilities.

## 5. How GenAI Influenced Our Research

ChatGPT influenced both our workflow and the direction of our investigation.

It helped us move from general classification performance to more specific questions about visual shortcuts and model behavior.

The prompt comparison experiment was particularly interesting because descriptive prompts produced worse accuracy and changed many predictions. This encouraged us to investigate how sensitive CLIP is to text descriptions.

AI assistance also helped us organize the results into an interactive dashboard, making it easier to explore individual errors alongside overall statistics.

However, the final decisions about which experiments to conduct and which findings to include remained ours.

## 6. What We Would Do Differently

If we repeated this project, we would define our research questions and evaluation criteria more clearly before implementing the experiments.

We would also organize our code and files earlier to avoid integration problems later.

When using ChatGPT, we would provide more context about our existing code and ask for smaller changes instead of large code blocks. This would make the suggestions easier to test and integrate.

Finally, we would document important AI interactions throughout the project rather than reconstructing them afterwards.

## 7. Conclusion

Using ChatGPT was helpful for our project, especially for coding, debugging, and exploring new analysis directions.

At the same time, we learned that AI-generated code and interpretations require careful verification. Some suggestions needed corrections, and not every proposed approach improved our results.

Our main takeaway is that **Generative AI is most useful as a collaborative assistant when we actively test its suggestions, question its explanations, and make our own research decisions.**

More detailed examples of our interactions with ChatGPT are documented in [`ai-interactions/phase1-ai-interactions.md`](ai-interactions/phase1-ai-interactions.md).
