def save_report(analysis_results, filepath="final_report.html"):
    with open(filepath, 'w') as file:
        file.write("<h1>Reporte Final</h1>")
        file.write(f"<p>{analysis_results}</p>")
