import { useRef, useState } from "react";
import { Send } from "lucide-react";
import ReactMarkdown from "react-markdown";

const API_URL = `${import.meta.env.VITE_KEYWORD_API}/predict_from_text`;

const Home = () => {
	const [chats, setChats] = useState([]);
	const [loading, setLoading] = useState(false);
	const messageRef = useRef(null);

	const handleSubmit = async (e) => {
		e.preventDefault();

		const message = messageRef.current.value.trim();
		if (!message) return;

		// Add the user's message to the chat
		setChats((prevChats) => [...prevChats, { sender: "user", message }]);
		messageRef.current.value = ""; // Clear textarea after submit

		setLoading(true);

		try {
			// Send the message to the API with 'text' as the key
			const response = await fetch(API_URL, {
				method: "POST",
				headers: {
					"Content-Type": "application/json", // Ensure JSON format
				},
				body: JSON.stringify({ text: message }), // Sending { text: message }
			});

			// Handle response from API
			const data = await response.json();
			console.log(data);

			// Check if API response contains parsedData and prediction
			if (data.formatted_output) {
				const formattedText = data.formatted_output.parts[0].text;
				setChats((prevChats) => [
					...prevChats,
					{
						sender: "api",
						message: formattedText,
					},
				]);
			} else {
				console.error("No parsedData in response");
			}
		} catch (error) {
			console.error("Error sending message:", error);
		} finally {
			setLoading(false); // Reset loading after API response
		}
	};

	const handleKeyDown = (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSubmit(e);
		}
	};

	return (
		<main className='relative h-[80vh] flex flex-col overflow-y-auto custom-scrollbar gap-2 pb-32'>
			{chats.length > 0 ? (
				chats.map((blob, index) => (
					<div
						key={index}
						className={`max-w-3/5 rounded-2xl p-4 ${
							blob.sender === "user" ? "bg-black ml-auto" : "bg-accent"
						}`}
					>
						<ReactMarkdown>{blob.message}</ReactMarkdown>
					</div>
				))
			) : (
				<p>Write a message to begin...</p>
			)}

			{/* Loading Indicator */}
			{loading && (
				<div className='max-w-3/5 rounded-2xl p-4 bg-accent'>
					<p>Loading...</p>
				</div>
			)}

			<form onSubmit={handleSubmit} className='fixed bottom-0 left-[10rem] py-12 w-3/4 mx-auto'>
				<textarea
					ref={messageRef}
					onKeyDown={handleKeyDown}
					className='w-full h-24 text-primary bg-foreground rounded-3xl p-4 px-6 resize-none text-xl outline-none'
					placeholder='Type a message...'
				/>
				<button type='submit' className='hidden'>
					<Send />
				</button>
			</form>
		</main>
	);
};

export default Home;
