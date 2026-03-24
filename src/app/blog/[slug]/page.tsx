import { notFound } from "next/navigation";
import Link from "next/link";
import { getAllPosts, getPost } from "@/lib/posts";
import { PostBody } from "./PostBody";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return getAllPosts().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) return {};
  return {
    title: `${post.title} — AI Text X-Ray Dev Log`,
    description: post.summary,
  };
}

export default async function BlogPost({ params }: Props) {
  const { slug } = await params;
  const post = getPost(slug);
  if (!post) notFound();

  return (
    <article>
      <Link
        href="/blog"
        className="text-xs text-[var(--muted)] hover:text-[var(--accent)] transition-colors"
      >
        &larr; All posts
      </Link>

      <header className="mt-4 mb-8">
        <div className="flex items-center gap-2 mb-2">
          <time className="text-xs text-[var(--muted)]">{post.date}</time>
          {post.tags.map((tag) => (
            <span
              key={tag}
              className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--accent-light)] text-[var(--accent)] font-medium"
            >
              {tag}
            </span>
          ))}
        </div>
        <h1 className="text-2xl font-bold text-[var(--foreground)] tracking-tight">
          {post.title}
        </h1>
        {post.summary && (
          <p className="text-sm text-[var(--muted)] mt-2">{post.summary}</p>
        )}
      </header>

      <PostBody content={post.content} />
    </article>
  );
}
