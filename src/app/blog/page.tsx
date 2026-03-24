import Link from "next/link";
import { getAllPosts } from "@/lib/posts";

export default function BlogIndex() {
  const posts = getAllPosts();

  return (
    <div>
      <h1 className="text-2xl font-bold text-[var(--foreground)] tracking-tight">
        Dev Log
      </h1>
      <p className="text-sm text-[var(--muted)] mt-1 mb-8">
        What we built, why, and how it works.
      </p>

      {posts.length === 0 ? (
        <p className="text-sm text-[var(--muted)]">No posts yet.</p>
      ) : (
        <div className="space-y-6">
          {posts.map((post) => (
            <article key={post.slug}>
              <Link
                href={`/blog/${post.slug}`}
                className="block bg-[var(--card)] rounded-xl border border-[var(--card-border)] p-5 hover:border-[var(--accent)]/40 transition-colors group"
              >
                <div className="flex items-center gap-2 mb-2">
                  <time className="text-xs text-[var(--muted)]">
                    {post.date}
                  </time>
                  {post.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--accent-light)] text-[var(--accent)] font-medium"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                <h2 className="text-base font-semibold text-[var(--foreground)] group-hover:text-[var(--accent)] transition-colors">
                  {post.title}
                </h2>
                {post.summary && (
                  <p className="text-sm text-[var(--muted)] mt-1 leading-relaxed">
                    {post.summary}
                  </p>
                )}
              </Link>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
