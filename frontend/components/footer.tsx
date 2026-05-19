export function Footer() {
  return (
    <footer className="border-t border-border bg-card mt-8">
      <div className="max-w-[1200px] mx-auto px-6 py-4 flex justify-between items-baseline text-xs text-muted-foreground">
        <span className="font-medium text-foreground">CineEmbed</span>
        <span className="tabular-nums">SENG 474 · TED University · Spring 2026</span>
      </div>
      <div className="border-t border-border/50">
        <div className="max-w-[1200px] mx-auto px-6 py-2 text-[10px] text-muted-foreground/70 tabular-nums text-center">
          329,044 films · 3 backbones · L2-normalized cosine · multimodal autoencoder over 7 feature blocks
        </div>
      </div>
    </footer>
  );
}
