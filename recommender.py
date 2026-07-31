def build_recommendations(profile):
    recommendations = []
    if profile.programming_skill is None or profile.programming_skill < 6:
        recommendations.append("Practice coding problems and strengthen data structures & algorithms.")
    if profile.communication_skill is None or profile.communication_skill < 6:
        recommendations.append("Take mock interviews and improve your communication skills.")
    if profile.aptitude_score is None or profile.aptitude_score < 60:
        recommendations.append("Solve real-world problems and build one project end-to-end.")
    if profile.internships is None or profile.internships < 1:
        recommendations.append("Look for internships or part-time industry exposure.")
    if profile.projects is None or profile.projects < 2:
        recommendations.append("Build at least two demonstrable projects and add them to your profile.")
    if not recommendations:
        recommendations.append("You are on a strong track. Keep updating your skills and tracking progress.")
    return recommendations
