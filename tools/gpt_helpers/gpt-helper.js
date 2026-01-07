import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// 🧠 Function: Ask GPT about a file
async function explainFile(filePath) {
  const fullPath = path.resolve(filePath);
  const code = fs.readFileSync(fullPath, "utf-8");

  const response = await client.chat.completions.create({
    model: "gpt-4.1",
    messages: [
      { role: "system", content: "You are a helpful coding assistant that can read and write code." },
      { role: "user", content: `Explain the following code:\n\n${code}` }
    ]
  });

  console.log(`\n📝 Explanation for ${filePath}:\n`);
  console.log(response.choices[0].message.content);
}

// 🧠 Function: Modify a file with GPT
async function rewriteFile(filePath, instruction) {
  const fullPath = path.resolve(filePath);
  const code = fs.readFileSync(fullPath, "utf-8");

  const response = await client.chat.completions.create({
    model: "gpt-4.1",
    messages: [
      { role: "system", content: "You are a coding assistant that edits files according to user instructions." },
      { role: "user", content: `Here is the file content:\n\n${code}` },
      { role: "user", content: `Please modify it according to this instruction: ${instruction}` }
    ]
  });

  const newCode = response.choices[0].message.content;
  fs.writeFileSync(fullPath, newCode);
  console.log(`✅ File updated: ${filePath}`);
}

// CLI handling
const [,, command, file, ...rest] = process.argv;
const instruction = rest.join(" ");

if (command === "explain") {
  await explainFile(file);
} else if (command === "rewrite") {
  await rewriteFile(file, instruction);
} else {
  console.log(`Usage:
  node gpt-helper.js explain <filename>
  node gpt-helper.js rewrite <filename> "<instruction>"
  `);
}
