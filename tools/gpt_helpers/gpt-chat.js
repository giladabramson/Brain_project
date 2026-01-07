import OpenAI from "openai";
import readline from "readline";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Set up terminal input/output
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: '🧠 GPT> '
});

let conversation = [
  { role: "system", content: "You are a helpful coding assistant working directly in the user's terminal." }
];

// Function to send a message to GPT
async function sendToGPT(message) {
  conversation.push({ role: "user", content: message });
  const response = await client.chat.completions.create({
    model: "gpt-4.1",
    messages: conversation
  });

  const reply = response.choices[0].message.content;
  conversation.push({ role: "assistant", content: reply });
  console.log(`\n🤖 ${reply}\n`);
  rl.prompt();
}

// Start the chat loop
console.log("🚀 Terminal Chat started. Type your messages below. Type 'exit' to quit.\n");
rl.prompt();

rl.on('line', async (line) => {
  const trimmed = line.trim();
  if (trimmed.toLowerCase() === 'exit') {
    rl.close();
    return;
  }
  await sendToGPT(trimmed);
}).on('close', () => {
  console.log('👋 Chat ended.');
  process.exit(0);
});
