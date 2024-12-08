import os
import pytesseract
import pdf2image
from bson.objectid import ObjectId
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph


class PDFUtils:
    @staticmethod
    def clean_text_for_pdf(text):
        text = text.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u201C", '"').replace(u"\u201D", '"').replace(
            u"\u2013", '"')
        return text

    @staticmethod
    def create_pdf_for_ai_review_results_enhanced(ai_review_results, zcases_collection):
        save_dir = "enhanced_ai_review_results"
        os.makedirs(save_dir, exist_ok=True)

        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        heading_style = styles['Heading1']
        subheading_style = styles['Heading2']

        for result in ai_review_results:
            this_zcase = zcases_collection.find_one({'_id': ObjectId(result.get('zcase_id'))})
            if this_zcase is None:
                print(f"ZCase with ID {result.get('zcase_id')} not found.")
                continue

            pdf_file_name = os.path.join(save_dir, f"{this_zcase.get('_id')}.pdf")
            doc = SimpleDocTemplate(pdf_file_name, pagesize=letter)
            story = []

            story.append(Paragraph("Case Information", heading_style))
            story.append(Spacer(1, 12))

            this_volume = this_zcase.get('volume')
            this_reporter = this_zcase.get('reporter')
            case_info_content = f"""
            <b>Case ID:</b> {this_zcase.get('_id')}<br/>
            <b>Volume Number:</b> {this_volume.get('volume_number')}<br/>
            <b>Reporter:</b> {this_reporter.get('full_name')}<br/>
            <b>First Page:</b> {this_zcase.get('first_page')}<br/>
            <b>Checked for Case:</b> {this_zcase.get('checked_for_case')}<br/>
            <b>Case Found:</b> {this_zcase.get('case_found')}
            """
            story.append(Paragraph(case_info_content, normal_style))
            story.append(Spacer(1, 12))

            story.append(Paragraph("AI Review Requests", subheading_style))
            for ai_review_result in this_zcase.get('ai_review_requests'):
                result_content = f"""
                <b>Request Type:</b> {ai_review_result.get('request_type')}<br/>
                <b>Content Field:</b> {ai_review_result.get('page_content_key')}<br/>
                <b>Status:</b> {ai_review_result.get('status')}<br/>
                <b>Requested On:</b> {ai_review_result.get('requested_on')}<br/>
                <b>AI Review Result:</b><br/>
                {ai_review_result.get('ai_review_result')}
                """
                story.append(Paragraph(result_content, normal_style))
                story.append(Spacer(1, 12))

            doc.build(story)
            print(f"PDF for AI review result enhanced version saved to {pdf_file_name}")

    @staticmethod
    def ocr_pdf_to_text(pdf_path):
        try:
            pages = pdf2image.convert_from_path(pdf_path)
            text = ''.join([pytesseract.image_to_string(page) for page in pages])
            return text
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""
