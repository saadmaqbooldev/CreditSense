import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CreditSense | Underwriting Console",
  description: "Evidence-backed SME credit underwriting workspace",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return <html lang="en"><body>{children}</body></html>;
}
