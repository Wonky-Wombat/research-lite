#
# excel_loader.py
# ResearchLite
#
# Created by Wonky-Wombat on 2026-09-22.
#

import datetime

from langchain_core.documents import Document

from ..utils.loader_utils import build_source_metadata, iter_files, populate_document_metadata


def load_excel(path: str) -> list[Document]:
    """Load Excel workbooks from a file or directory, one document per data row."""
    from python_calamine import CalamineWorkbook

    documents: list[Document] = []
    for excel_file in iter_files(path, extensions=["xlsx", "xls"]):
        source_metadata = build_source_metadata(excel_file)
        workbook = CalamineWorkbook.from_path(str(excel_file))
        rows_loaded = 0
        for sheet_name in workbook.sheet_names:
            rows = iter(workbook.get_sheet_by_name(sheet_name).to_python(skip_empty_area=False))
            header = [_cell_text(cell) for cell in next(rows, [])]
            for row in rows:
                values = [_cell_text(cell) for cell in row]
                if not any(values):
                    continue
                lines = [f"sheet: {sheet_name}"] + [
                    f"{_column_name(header, index)}: {value}"
                    for index, value in enumerate(values)
                    if value
                ]
                document = Document(page_content="\n".join(lines))
                documents.append(
                    populate_document_metadata(
                        document, source_metadata, source_unit_index=rows_loaded
                    )
                )
                rows_loaded += 1
    return documents


def _column_name(header: list[str], index: int) -> str:
    name = header[index] if index < len(header) else ""
    return name or f"column {index + 1}"


def _cell_text(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, datetime.datetime) and value.time() == datetime.time():
        value = value.date()
    if isinstance(value, datetime.date | datetime.time):
        return value.isoformat()
    return "" if value is None else str(value).strip()
