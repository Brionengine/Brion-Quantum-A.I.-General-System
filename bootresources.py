def bootstrap_medical_data():
    """
    Bootstraps medical data and necessary resources for analysis.
    """
    print("Bootstrapping medical data...")

    # Pre-load necessary medical models
    ryan.add_medical_knowledge("Cardiology", "Knowledge about heart diseases")
    ryan.add_medical_knowledge("Neurology", "Insights on brain function and disorders")

    print("Medical data pre-loaded for analysis.")

bootstrap_medical_data()
