from pathlib import Path
from docx import Document


def remove_content_controls(in_path, out_path=None):
    """
    Remove TODOS os content controls (w:sdt) de um .docx,
    preservando o conteúdo interno (w:sdtContent).
    """
    in_path = Path(in_path)

    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_sem_controles.docx")

    doc = Document(in_path)
    root = doc.element  # elemento raiz do documento

    # Pega todos os <w:sdt> ignorando namespace (usa local-name())
    sdts = root.xpath(".//*[local-name()='sdt']")

    for sdt in sdts:
        parent = sdt.getparent()

        # Pega o <w:sdtContent> diretamente abaixo do <w:sdt>
        content_elems = sdt.xpath("./*[local-name()='sdtContent']")

        if content_elems:
            content = content_elems[0]
            idx = list(parent).index(sdt)

            # Insere o conteúdo do sdtContent no lugar do sdt
            for child in list(content):
                parent.insert(idx, child)
                idx += 1

        # Remove o próprio <w:sdt> (com ou sem conteúdo)
        parent.remove(sdt)

    doc.save(out_path)
    print(f"Salvo em: {out_path}")


if __name__ == "__main__":
    # Ajusta aqui o nome do arquivo de entrada
    remove_content_controls(
        "Supplementary_Methods_short.docx",
        "Supplementary_Methods_short_sem_controles.docx",
    )
