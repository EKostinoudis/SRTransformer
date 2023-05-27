build_data_files_report = report.aux  report.bcf  report.log  report.run.xml  report.toc report.blg report.bbl

report: report.tex
	xelatex $? && biber report && xelatex $?

clean:
	rm ${build_data_files_report}

