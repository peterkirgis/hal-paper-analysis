"""
Report Generator for HAL Benchmark Analysis

This module provides functionality to generate comprehensive PDF reports
summarizing benchmark analysis results, including sanity check failures
and evaluation failures across different models and benchmarks.
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF


class BenchmarkReportGenerator:
    """
    A comprehensive report generator for benchmark analysis results.
    
    This class creates detailed PDF reports with statistics, charts, and
    summaries of sanity check failures and evaluation failures.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir (str): Directory to save generated reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=5
        )
        
        # Subheading style
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=15,
            textColor=colors.darkgreen
        )
        
        # Summary box style
        self.summary_style = ParagraphStyle(
            'SummaryBox',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            spaceBefore=10,
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=10,
            backColor=colors.lightgrey
        )
        
        # Alert style for failures
        self.alert_style = ParagraphStyle(
            'AlertBox',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            spaceBefore=10,
            leftIndent=20,
            rightIndent=20,
            borderWidth=2,
            borderColor=colors.red,
            borderPadding=10,
            backColor=colors.mistyrose
        )
        
        # Success style
        self.success_style = ParagraphStyle(
            'SuccessBox',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            spaceBefore=10,
            leftIndent=20,
            rightIndent=20,
            borderWidth=2,
            borderColor=colors.green,
            borderPadding=10,
            backColor=colors.lightgreen
        )

    def create_summary_table(self, data: List[List[str]], headers: List[str]) -> Table:
        """
        Create a formatted table for summary data.
        
        Args:
            data (List[List[str]]): Table data rows
            headers (List[str]): Table headers
            
        Returns:
            Table: Formatted ReportLab table
        """
        # Combine headers and data
        table_data = [headers] + data
        
        # Create table
        table = Table(table_data, repeatRows=1)
        
        # Style the table
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            
            # Data row styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        return table

    def create_success_rate_chart(self, model_success_stats: Dict[str, Dict], chart_title: str) -> Drawing:
        """
        Create a bar chart showing success rates by model.
        
        Args:
            model_success_stats (Dict[str, Dict]): Model success statistics
            chart_title (str): Title for the chart
            
        Returns:
            Drawing: ReportLab drawing with bar chart
        """
        if not model_success_stats:
            # Create empty chart placeholder
            drawing = Drawing(400, 200)
            drawing.add(Rect(0, 0, 400, 200, fillColor=colors.lightgrey, strokeColor=colors.grey))
            return drawing
        
        drawing = Drawing(500, 300)
        
        # Create bar chart
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 200
        chart.width = 400
        
        # Prepare data - sort by success rate
        sorted_models = sorted(model_success_stats.items(), 
                              key=lambda x: x[1].get('success_rate', 0), reverse=True)
        
        models = [model[:20] + "..." if len(model) > 20 else model for model, _ in sorted_models]
        success_rates = [stats.get('success_rate', 0) for _, stats in sorted_models]
        
        chart.data = [success_rates]
        chart.categoryAxis.categoryNames = models
        chart.categoryAxis.labels.angle = 45
        chart.categoryAxis.labels.fontSize = 8
        
        # Styling - use green for success rates
        chart.bars[0].fillColor = colors.green
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = 100
        chart.valueAxis.valueStep = 20
        
        drawing.add(chart)
        return drawing

    def create_failure_chart(self, failure_data: Dict[str, int], chart_title: str) -> Drawing:
        """
        Create a bar chart showing failure counts by model.
        
        Args:
            failure_data (Dict[str, int]): Model name to failure count mapping
            chart_title (str): Title for the chart
            
        Returns:
            Drawing: ReportLab drawing with bar chart
        """
        if not failure_data:
            # Create empty chart placeholder
            drawing = Drawing(400, 200)
            drawing.add(Rect(0, 0, 400, 200, fillColor=colors.lightgrey, strokeColor=colors.grey))
            return drawing
        
        drawing = Drawing(500, 300)
        
        # Create bar chart
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 200
        chart.width = 400
        
        # Prepare data
        models = list(failure_data.keys())
        failures = list(failure_data.values())
        
        chart.data = [failures]
        chart.categoryAxis.categoryNames = models
        chart.categoryAxis.labels.angle = 45
        chart.categoryAxis.labels.fontSize = 8
        
        # Styling
        chart.bars[0].fillColor = colors.red
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(failures) * 1.1 if failures else 1
        
        drawing.add(chart)
        return drawing

    def generate_benchmark_report(
        self,
        benchmark_name: str,
        sanity_check_failures: Dict[str, int],
        eval_failures: Dict[str, int],
        total_files_processed: int,
        total_runs_processed: int,
        processing_summary: Dict[str, Any],
        output_filename: str = None
    ) -> str:
        """
        Generate a comprehensive PDF report for benchmark analysis.
        
        Args:
            benchmark_name (str): Name of the benchmark
            sanity_check_failures (Dict[str, int]): Model to sanity failure count mapping
            eval_failures (Dict[str, int]): Model to eval failure count mapping
            total_files_processed (int): Total number of files processed
            total_runs_processed (int): Total number of agent runs processed
            processing_summary (Dict[str, Any]): Additional processing statistics
            output_filename (str): Custom filename for the report
            
        Returns:
            str: Path to the generated PDF report
        """
        # Generate filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{benchmark_name.lower().replace(' ', '_')}_report_{timestamp}.pdf"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story (content)
        story = []
        
        # Title
        story.append(Paragraph(f"{benchmark_name} Analysis Report", self.title_style))
        story.append(Spacer(1, 12))
        
        # Generation info
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated on: {generation_time}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.heading_style))
        
        total_sanity_failures = sum(sanity_check_failures.values())
        total_eval_failures = sum(eval_failures.values())
        
        summary_text = f"""
        <b>Files Processed:</b> {total_files_processed}<br/>
        <b>Agent Runs Processed:</b> {total_runs_processed}<br/>
        <b>Total Sanity Check Failures:</b> {total_sanity_failures}<br/>
        <b>Total Evaluation Failures:</b> {total_eval_failures}<br/>
        <b>Success Rate:</b> {((total_runs_processed - total_eval_failures) / total_runs_processed * 100):.1f}% if total_runs_processed > 0 else 'N/A'
        """
        
        if total_sanity_failures > 0 or total_eval_failures > 0:
            story.append(Paragraph(summary_text, self.alert_style))
        else:
            story.append(Paragraph(summary_text, self.success_style))
        
        story.append(Spacer(1, 20))
        
        # Sanity Check Failures Section
        story.append(Paragraph("Sanity Check Failures", self.heading_style))
        
        if sanity_check_failures:
            story.append(Paragraph(
                f"Found {total_sanity_failures} sanity check failures across {len(sanity_check_failures)} models:",
                self.styles['Normal']
            ))
            story.append(Spacer(1, 10))
            
            # Create table for sanity failures
            sanity_data = [[model, str(count)] for model, count in sanity_check_failures.items()]
            sanity_table = self.create_summary_table(
                sanity_data,
                ["Model", "Sanity Check Failures"]
            )
            story.append(sanity_table)
            story.append(Spacer(1, 15))
            
            # Add chart
            sanity_chart = self.create_failure_chart(sanity_check_failures, "Sanity Check Failures by Model")
            story.append(sanity_chart)
            
            # Add explanation
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                "<b>Note:</b> Sanity check failures indicate inconsistencies in the logging data where "
                "smaller logs are not proper subsets of larger logs for the same task.",
                self.styles['Normal']
            ))
        else:
            story.append(Paragraph(
                "✅ No sanity check failures detected! All logging data appears consistent.",
                self.success_style
            ))
        
        story.append(Spacer(1, 20))
        
        # Evaluation Failures Section
        story.append(Paragraph("Evaluation Failures", self.heading_style))
        
        if eval_failures:
            story.append(Paragraph(
                f"Found {total_eval_failures} evaluation failures across {len(eval_failures)} models:",
                self.styles['Normal']
            ))
            story.append(Spacer(1, 10))
            
            # Create table for eval failures
            eval_data = [[model, str(count)] for model, count in eval_failures.items()]
            eval_table = self.create_summary_table(
                eval_data,
                ["Model", "Evaluation Failures"]
            )
            story.append(eval_table)
            story.append(Spacer(1, 15))
            
            # Add chart
            eval_chart = self.create_failure_chart(eval_failures, "Evaluation Failures by Model")
            story.append(eval_chart)
            
            # Add explanation
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                "<b>Note:</b> Evaluation failures occur when tasks cannot be processed due to "
                "missing evaluation results or processing errors.",
                self.styles['Normal']
            ))
        else:
            story.append(Paragraph(
                "✅ No evaluation failures detected! All tasks were successfully processed.",
                self.success_style
            ))
        
        story.append(PageBreak())
        
        # Detailed Statistics
        story.append(Paragraph("Detailed Statistics", self.heading_style))
        
        # Processing summary
        if processing_summary:
            story.append(Paragraph("Processing Summary", self.subheading_style))
            
            for key, value in processing_summary.items():
                if key != "model_success_stats":  # Handle this separately
                    story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", self.styles['Normal']))
            
            story.append(Spacer(1, 15))
        
        # Model Success Statistics
        model_success_stats = processing_summary.get("model_success_stats", {}) if processing_summary else {}
        if model_success_stats:
            story.append(Paragraph("Model Success Statistics", self.subheading_style))
            
            # Create detailed success table
            success_data = []
            for model, stats in model_success_stats.items():
                success_data.append([
                    model,
                    str(stats.get("total_tasks_attempted", 0)),
                    str(stats.get("successful_runs", 0)),
                    str(stats.get("sanity_check_failures", 0)),
                    str(stats.get("eval_failures", 0)),
                    f"{stats.get('success_rate', 0):.1f}%"
                ])
            
            # Sort by success rate (highest first)
            success_data.sort(key=lambda x: float(x[5].replace('%', '')), reverse=True)
            
            success_table = self.create_summary_table(
                success_data,
                ["Model", "Tasks Attempted", "Successful Runs", "Sanity Failures", "Eval Failures", "Success Rate"]
            )
            story.append(success_table)
            story.append(Spacer(1, 15))
            
            # Add success rate chart
            success_chart = self.create_success_rate_chart(model_success_stats, "Model Success Rates")
            story.append(success_chart)
            story.append(Spacer(1, 10))
            
            # Add success rate insights
            if success_data:
                best_model = success_data[0]
                worst_model = success_data[-1]
                
                insights_text = f"""
                <b>Performance Insights:</b><br/>
                • Best performing model: <b>{best_model[0]}</b> ({best_model[5]} success rate)<br/>
                • Lowest performing model: <b>{worst_model[0]}</b> ({worst_model[5]} success rate)<br/>
                • Models with zero failures: <b>{len([m for m in success_data if m[3] == '0' and m[4] == '0'])}</b> out of {len(success_data)}
                """
                
                story.append(Paragraph(insights_text, self.summary_style))
                story.append(Spacer(1, 15))
        
        # Model Performance Overview
        if sanity_check_failures or eval_failures:
            story.append(Paragraph("Model Performance Overview", self.subheading_style))
            
            all_models = set(sanity_check_failures.keys()) | set(eval_failures.keys())
            
            performance_data = []
            for model in sorted(all_models):
                sanity_fails = sanity_check_failures.get(model, 0)
                eval_fails = eval_failures.get(model, 0)
                total_fails = sanity_fails + eval_fails
                
                performance_data.append([
                    model,
                    str(sanity_fails),
                    str(eval_fails),
                    str(total_fails)
                ])
            
            performance_table = self.create_summary_table(
                performance_data,
                ["Model", "Sanity Failures", "Eval Failures", "Total Failures"]
            )
            story.append(performance_table)
        
        # Build PDF
        doc.build(story)
        
        return output_path

    def generate_quick_summary_report(
        self,
        benchmark_name: str,
        sanity_failures: int,
        eval_failures: int,
        total_files: int,
        total_runs: int,
        output_filename: str = None
    ) -> str:
        """
        Generate a quick summary report with key metrics.
        
        Args:
            benchmark_name (str): Name of the benchmark
            sanity_failures (int): Total sanity check failures
            eval_failures (int): Total evaluation failures
            total_files (int): Total files processed
            total_runs (int): Total runs processed
            output_filename (str): Custom filename
            
        Returns:
            str: Path to the generated PDF report
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{benchmark_name.lower().replace(' ', '_')}_summary_{timestamp}.pdf"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        story.append(Paragraph(f"{benchmark_name} - Quick Summary", self.title_style))
        story.append(Spacer(1, 20))
        
        # Key metrics
        success_rate = ((total_runs - eval_failures) / total_runs * 100) if total_runs > 0 else 0
        
        metrics_text = f"""
        <b>Files Processed:</b> {total_files}<br/>
        <b>Runs Processed:</b> {total_runs}<br/>
        <b>Sanity Check Failures:</b> {sanity_failures}<br/>
        <b>Evaluation Failures:</b> {eval_failures}<br/>
        <b>Success Rate:</b> {success_rate:.1f}%
        """
        
        style = self.success_style if (sanity_failures == 0 and eval_failures == 0) else self.alert_style
        story.append(Paragraph(metrics_text, style))
        
        # Generation time
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        ))
        
        doc.build(story)
        return output_path


# Convenience functions for easy integration
def generate_benchmark_report(
    benchmark_name: str,
    sanity_check_failures: Dict[str, int],
    eval_failures: Dict[str, int] = None,
    total_files_processed: int = 0,
    total_runs_processed: int = 0,
    processing_summary: Dict[str, Any] = None,
    output_dir: str = "reports"
) -> str:
    """
    Convenience function to generate a benchmark report.
    
    Args:
        benchmark_name (str): Name of the benchmark
        sanity_check_failures (Dict[str, int]): Model to sanity failure count mapping
        eval_failures (Dict[str, int]): Model to eval failure count mapping
        total_files_processed (int): Total files processed
        total_runs_processed (int): Total runs processed
        processing_summary (Dict[str, Any]): Additional processing statistics
        output_dir (str): Output directory for reports
        
    Returns:
        str: Path to generated report
    """
    generator = BenchmarkReportGenerator(output_dir)
    
    return generator.generate_benchmark_report(
        benchmark_name=benchmark_name,
        sanity_check_failures=sanity_check_failures,
        eval_failures=eval_failures or {},
        total_files_processed=total_files_processed,
        total_runs_processed=total_runs_processed,
        processing_summary=processing_summary or {}
    )


def generate_quick_summary(
    benchmark_name: str,
    sanity_failures: int,
    eval_failures: int,
    total_files: int,
    total_runs: int,
    output_dir: str = "reports"
) -> str:
    """
    Convenience function to generate a quick summary report.
    
    Args:
        benchmark_name (str): Name of the benchmark
        sanity_failures (int): Total sanity failures
        eval_failures (int): Total eval failures
        total_files (int): Total files processed
        total_runs (int): Total runs processed
        output_dir (str): Output directory
        
    Returns:
        str: Path to generated report
    """
    generator = BenchmarkReportGenerator(output_dir)
    
    return generator.generate_quick_summary_report(
        benchmark_name=benchmark_name,
        sanity_failures=sanity_failures,
        eval_failures=eval_failures,
        total_files=total_files,
        total_runs=total_runs
    )
