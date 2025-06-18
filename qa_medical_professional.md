# Q&A for Medical Professionals: TAGT Project

## Clinical and Practical Questions

**Q1: What is TAGT and how can it help my patients?**
A: TAGT is an AI model that predicts if a patient with Systemic Lupus Erythematosus (SLE) will have a flare soon. It uses patient gene data, protein interactions, and clinical scores to give early warnings, so you can intervene before the patient's condition worsens.

**Q2: What data does TAGT use?**
A: It uses gene expression profiles, protein-protein interaction networks, and clinical parameters like SLEDAI scores, all tracked over time.

**Q3: How accurate is TAGT?**
A: TAGT achieves 96.3% AUC-ROC, which means it is very good at distinguishing between patients who will and will not have a flare.

**Q4: How does TAGT improve patient care?**
A: By predicting flares early, it allows for timely treatment, personalized care, and can help prevent severe complications and organ damage.

**Q5: Is TAGT easy to use in a hospital setting?**
A: The goal is to make TAGT available as a simple software tool that integrates with hospital systems. It will provide clear risk scores and explanations for each patient.

**Q6: What are the limitations?**
A: TAGT needs good quality data from multiple sources. It may not work as well if some data is missing or incorrect. Also, it is still being validated for use in different hospitals.

**Q7: How does TAGT explain its predictions?**
A: The model can highlight which factors (genes, proteins, clinical scores) were most important for its prediction, helping you understand the reasoning.

## Technical Questions (Explained Simply)

**Q8: How does the model learn?**
A: TAGT uses machine learning to find patterns in past patient data. It learns which changes in genes, proteins, or clinical scores usually come before a flare.

**Q9: What is a Graph Neural Network?**
A: It's a type of AI that understands how proteins interact with each other, like a map of connections. This helps the model see how one protein's change can affect others.

**Q10: What is Temporal Attention?**
A: This is a method where the model pays more attention to important times in a patient's history, like just before a flare.

## Future Work and Goals

**Q11: What are the future plans for TAGT?**
A: We want to:
- Test TAGT in more hospitals and with more patients
- Make the model explain its predictions even better
- Add more types of data, like imaging or lifestyle factors
- Build a user-friendly tool for doctors

**Q12: How will these goals be achieved?**
A: By working with hospitals, collecting more data, and improving the software for easy use in clinics.

## In Simple Indian English

TAGT is a smart computer tool that helps doctors know if SLE patients will get worse soon. It looks at many things together—genes, proteins, and health scores over time. It gives early warning, so you can treat patients before they get serious problems. In future, we want to make it even better and easy for all doctors to use. 