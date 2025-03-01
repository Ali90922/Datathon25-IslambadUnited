import { useRef, useState } from "react";
import { Send } from "lucide-react";

const handleSubmit = (e) => {
	e.preventDefault();
};

const Home = () => {
	const [chats, setChats] = useState([]);
	const messageRef = useRef(null);

	const handleSubmit = (e) => {
		e.preventDefault();

		const message = messageRef.current.value.trim();
		if (!message) return;

		setChats([...chats, { sender: "user", message }]);
		messageRef.current.value = ""; // Clear textarea after submit
	};

	const handleKeyDown = (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSubmit(e);
		}
	};

	return (
		<main className='relative h-[80vh] flex flex-col gap-4 overflow-y-auto custom-scrollbar'>
			{chats.length > 0 ? (
				chats.map((blob, index) => (
					<div
						key={index}
						className={`max-w-3/5 rounded-2xl p-4 ${
							blob.sender === "user" ? "bg-black ml-auto" : "bg-secondary"
						}`}
					>
						<p>{blob.message}</p>
					</div>
				))
			) : (
				<p>Write a message to begin...</p>
			)}
			<form onSubmit={handleSubmit} className='relative py-12 flex'>
				<textarea
					ref={messageRef}
					onKeyDown={handleKeyDown}
					className='w-full h-24 bg-foreground rounded-3xl p-4 px-6 resize-none text-xl outline-none'
					placeholder='Type a message...'
				/>
				<button
					type='submit'
					className='ml-4 bg-accent text-white px-4 py-2 rounded-2xl flex items-center justify-center h-full aspect-square'
				>
					<Send />
				</button>
			</form>
		</main>
	);
};

export default Home;
