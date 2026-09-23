import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "ML Playground",
  description: "Train models on your own CSVs and compare runs.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>
        <nav>
          <strong>ML Playground</strong>
          <Link href="/">Datasets</Link>
          <Link href="/runs/">Runs</Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
