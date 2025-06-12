"use client"
import { useState } from "react"
import { SpacyNLPResult
} from "@/interfaces/nlpresult "

export default function Home() {
  const [text, setText] = useState("")
  const [result, setResult] = useState<SpacyNLPResult
  | null>(null);

  const analyze = async () => {
    const formData = new FormData()
    formData.append("text", text)

    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/analyzespacy`, {
      method: "POST",
      body: formData,
    })

    const data = await res.json()
    setResult(data)
  }

  return (
    <main className="p-4">
      <h1 className="text-2xl font-bold mb-4">Text Analyzer</h1>
      <textarea
        className="w-full p-2 border rounded mb-4"
        rows={6}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste your text here..."
      />
      <button onClick={analyze} className="bg-blue-600 text-white px-4 py-2 rounded">
        Analyze
      </button>
      {result && (
        <div className="mt-4">
          <h2 className="font-semibold">Summary:</h2>
          <p>{result.summary}</p>
          <h2 className="font-semibold mt-2">Keywords:</h2>
          <ul>
            {result.keywords.map((word: string, i: number) => (
              <li key={i}>• {word}</li>
            ))}
          </ul>
          <h2 className="font-semibold mt-2">Sentiment:</h2>
          <ul>
            {result.entities.map(({ text, label }, i: number) => (
              <li key={i}>• {text +' - '+ label}</li>
            ))}
          </ul>
        </div>
      )}
    </main>
  )
}
