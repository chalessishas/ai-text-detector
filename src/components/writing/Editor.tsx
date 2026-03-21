"use client";

import { useEffect, useCallback, useImperativeHandle, forwardRef } from "react";
import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { Extension } from "@tiptap/react";
import type { Annotation, Trait } from "@/lib/writing/types";

const TRAIT_COLORS: Record<Trait, string> = {
  ideas: "#3b82f6",
  organization: "#a855f7",
  voice: "#f97316",
  wordChoice: "#14b8a6",
  fluency: "#22c55e",
  conventions: "#6b7280",
  presentation: "#ec4899",
};

export interface EditorHandle {
  scrollToAnnotation: (id: string) => void;
}

interface EditorProps {
  content: string;
  onUpdate: (html: string) => void;
  annotations: Annotation[];
  onAnnotationClick: (annotationId: string) => void;
  disabled?: boolean;
}

const annotationPluginKey = new PluginKey("annotationHighlight");

function buildDecorations(
  doc: ReturnType<typeof Object>,
  annotations: Annotation[]
): DecorationSet {
  const decorations: Decoration[] = [];
  if (!annotations.length) return DecorationSet.create(doc, []);

  const paragraphs: { node: ReturnType<typeof Object>; pos: number }[] = [];
  doc.descendants((node: ReturnType<typeof Object>, pos: number) => {
    if (node.isBlock && node.isTextblock) {
      paragraphs.push({ node, pos });
    }
  });

  for (const ann of annotations) {
    const para = paragraphs[ann.paragraph];
    if (!para) continue;

    const paraStart = para.pos + 1;
    const paraLen = para.node.content.size;

    let from: number;
    let to: number;

    if (
      ann.startOffset === -1 ||
      ann.endOffset === -1 ||
      ann.startOffset >= paraLen ||
      ann.endOffset > paraLen
    ) {
      from = paraStart;
      to = paraStart + paraLen;
    } else {
      from = paraStart + ann.startOffset;
      to = paraStart + ann.endOffset;
    }

    if (from >= to) continue;

    const color = TRAIT_COLORS[ann.trait];
    decorations.push(
      Decoration.inline(from, to, {
        style: `background-color: ${color}33; border-bottom: 2px solid ${color}; cursor: pointer;`,
        "data-annotation-id": ann.id,
        class: "annotation-highlight",
      })
    );
  }

  return DecorationSet.create(doc, decorations);
}

function createAnnotationExtension(annotations: Annotation[]) {
  return Extension.create({
    name: "annotationHighlight",

    addProseMirrorPlugins() {
      return [
        new Plugin({
          key: annotationPluginKey,
          state: {
            init(_, { doc }) {
              return buildDecorations(doc, annotations);
            },
            apply(tr, oldDecorations) {
              if (tr.docChanged || tr.getMeta(annotationPluginKey)) {
                return buildDecorations(tr.doc, annotations);
              }
              return oldDecorations.map(tr.mapping, tr.doc);
            },
          },
          props: {
            decorations(state) {
              return this.getState(state) ?? DecorationSet.empty;
            },
          },
        }),
      ];
    },
  });
}

const Editor = forwardRef<EditorHandle, EditorProps>(function Editor(
  { content, onUpdate, annotations, onAnnotationClick, disabled = false },
  ref
) {
  const editor = useEditor({
    extensions: [StarterKit, createAnnotationExtension(annotations)],
    content,
    editable: !disabled,
    onUpdate: ({ editor: e }) => {
      onUpdate(e.getHTML());
    },
    editorProps: {
      attributes: {
        class:
          "prose prose-sm max-w-none focus:outline-none min-h-[300px] p-4 text-[var(--foreground)]",
      },
    },
  });

  useImperativeHandle(ref, () => ({
    scrollToAnnotation(id: string) {
      const el = editor?.view.dom.querySelector(
        `[data-annotation-id="${id}"]`
      );
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("annotation-pulse");
      setTimeout(() => el.classList.remove("annotation-pulse"), 1500);
    },
  }), [editor]);

  // Update decorations when annotations change
  useEffect(() => {
    if (!editor) return;
    const { tr } = editor.state;
    tr.setMeta(annotationPluginKey, true);
    editor.view.dispatch(tr);
  }, [annotations, editor]);

  // Sync content from parent if it changes externally
  useEffect(() => {
    if (!editor) return;
    if (editor.getHTML() !== content && content !== undefined) {
      editor.commands.setContent(content, { emitUpdate: false });
    }
  }, [content, editor]);

  // Click handler for annotation highlights
  const handleClick = useCallback(
    (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      const annotationEl = target.closest("[data-annotation-id]");
      if (annotationEl) {
        const id = annotationEl.getAttribute("data-annotation-id");
        if (id) onAnnotationClick(id);
      }
    },
    [onAnnotationClick]
  );

  useEffect(() => {
    const el = editor?.view.dom;
    if (!el) return;
    el.addEventListener("click", handleClick);
    return () => el.removeEventListener("click", handleClick);
  }, [editor, handleClick]);

  return (
    <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-xl overflow-hidden">
      <EditorContent editor={editor} />
    </div>
  );
});

export default Editor;
