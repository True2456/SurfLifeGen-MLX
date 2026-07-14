class Surflifegen < Formula
  include Language::Python::Virtualenv

  desc "Apple Silicon native MLX 8-Bit synthetic aerial dataset generator & YOLO annotator"
  homepage "https://github.com/True2456/SurfLifeGen-MLX"
  url "https://github.com/True2456/SurfLifeGen-MLX.git", branch: "main"
  version "1.1.0"
  license "MIT"

  depends_on "python@3.11"
  depends_on :macos
  depends_on arch: :arm64 # Apple Silicon required for native MLX

  def install
    virtualenv_install_with_resources
  end

  test do
    system "#{bin}/surflifegen", "--help"
  end
end
