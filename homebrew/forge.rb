class Forge < Formula
  desc "Unified model merging, quantization & evaluation for Apple Silicon"
  homepage "https://github.com/shadyuwugurl/forge"
  url "https://github.com/shadyuwugurl/forge/archive/refs/tags/v0.5.0.tar.gz"
  sha256 "REPLACE_WITH_TARBALL_SHA256"
  license "MIT"
  depends_on "rust" => :build

  def install
    system "cargo", "install", "--locked", "--path", "crates/forge-cli", "--root", prefix
    # also install tui binary
    system "cargo", "install", "--locked", "--path", "crates/forge-tui", "--root", prefix
  end

  test do
    assert_match "forge", shell_output("#{bin}/forge --help")
  end
end
