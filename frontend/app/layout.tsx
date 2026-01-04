import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "Narrative Pulse",
  description: "Real-time crypto sentiment analytics",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased bg-black" suppressHydrationWarning>
        <Navbar />
        {children}
      </body>
    </html>
  );
}
