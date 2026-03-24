import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const jsonLdData = {
  "@context": "https://schema.org",
  "@type": "WebApplication",
  name: "AI Text X-Ray",
  description:
    "Free AI content detector that analyzes perplexity, GLTR token ranks, entropy, and burstiness to determine if text is AI-generated.",
  applicationCategory: "UtilitiesApplication",
  operatingSystem: "Any",
  offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
  featureList: [
    "AI text detection",
    "Perplexity analysis",
    "GLTR token rank visualization",
    "Entropy and burstiness measurement",
    "Sentence-level AI scoring",
    "AI text humanization",
    "Writing prompt generation",
  ],
};

function JsonLd() {
  // Static hardcoded JSON — no user input, safe to inline
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLdData) }}
    />
  );
}

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});


export const metadata: Metadata = {
  title: "AI Text X-Ray — Free AI Text Detector & Humanizer",
  description:
    "Free AI content detector that analyzes perplexity, GLTR token ranks, entropy, and burstiness to determine if text is AI-generated. Includes humanizer and writing tools.",
  keywords: [
    "AI text detector",
    "AI content detector",
    "ChatGPT detector",
    "AI writing detector",
    "perplexity checker",
    "GLTR",
    "AI humanizer",
    "detect AI text",
    "GPT detector free",
  ],
  openGraph: {
    title: "AI Text X-Ray — Free AI Text Detector & Humanizer",
    description:
      "Analyze text with perplexity, GLTR, entropy, and burstiness metrics. Detect AI-generated content and humanize it.",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "AI Text X-Ray — Free AI Text Detector",
    description:
      "Free explainable AI text detection with visual analysis. Perplexity, GLTR, entropy, burstiness metrics.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <JsonLd />
        {children}
      </body>
    </html>
  );
}
