import { useRef, useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { motion } from "framer-motion";

const API_URL = `${import.meta.env.VITE_BACKEND}/predict_from_text`;

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

	useEffect(() => {
		window.scrollTo({
			top: document.body.scrollHeight,
			behavior: "smooth",
		});
	}, [chats]);

	return (
		<>
			<section className='flex flex-col flex-grow overflow-y-auto custom-scrollbar gap-2 p-8 sm:px-16 md:px-32 lg:px-64'>
				{chats.length > 0 ? (
					chats.map((blob, index) => (
						<motion.div
							key={index}
							initial={{ scale: 0.9, opacity: 0 }}
							animate={{ scale: 1, opacity: 1 }}
							transition={{ duration: 1, ease: "easeOut" }}
							className={`rounded-2xl p-4 flex flex-col gap-4 max-w-11/12 ${
								blob.sender === "user" ? "bg-black ml-auto text-lg" : "bg-none text-xl pb-8"
							}`}
						>
							<ReactMarkdown>{blob.message}</ReactMarkdown>
						</motion.div>
					))
				) : (
					<p className='text-xl text-center'>
						Ask Substance Sense something! (e.g. "What is the overdose probability for a 23 year old
						male in Downtown Winnipeg?")
					</p>
				)}

				{/* Loading Indicator */}
				{loading && (
					<motion.div
						initial={{ opacity: 0, scale: 0.95 }}
						animate={{
							opacity: 1,
							scale: [1, 1.05, 1],
							transition: {
								duration: 1.5,
								repeat: Infinity,
								repeatType: "reverse",
								ease: "easeInOut",
							},
						}}
						exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.3 } }}
						className='max-w-3/5 rounded-2xl p-4 text-xl pb-8'
					>
						<p>Grabbing information...</p>
					</motion.div>
				)}
			</section>
			<form
				onSubmit={handleSubmit}
				className='fixed -bottom-2 lg:bottom-0 left-0 w-full lg:px-64 z-50 bg-gradient-to-t from-background to-transparent backdrop-blur-md'
			>
				<textarea
					ref={messageRef}
					onKeyDown={handleKeyDown}
					className='w-full text-primary bg-foreground rounded-t-3xl lg:rounded-3xl p-4 pb-16 px-6 resize-none text-lg outline-none'
					placeholder='Type a message...'
				/>
				<div className='w-full lg:py-4' />
			</form>
		</>
	);
};

export default Home;
