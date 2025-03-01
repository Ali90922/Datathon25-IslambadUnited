const handleSubmit = (e) => {
	e.preventDefault();
};

const Home = () => {
	const chats = [
		{
			sender: "user",
			message:
				"Lorem ipsum dolor sit amet consectetur, adipisicing elit. Praesentium, id adipisci sequi veritatis ad aliquid repellendus eius, quisquam laboriosam rerum ea mollitia doloribus inventore quod vero? Sunt, sed! Ea, commodi.",
		},
		{
			sender: "bot",
			message:
				"Lorem ipsum dolor sit amet consectetur, adipisicing elit. Praesentium, id adipisci sequi veritatis ad aliquid repellendus eius, quisquam laboriosam rerum ea mollitia doloribus inventore quod vero? Sunt, sed! Ea, commodi.",
		},
		{
			sender: "bot",
			message:
				"Lorem ipsum dolor sit amet consectetur, adipisicing elit. Praesentium, id adipisci sequi veritatis ad aliquid repellendus eius, quisquam laboriosam rerum ea mollitia doloribus inventore quod vero? Sunt, sed! Ea, commodi.",
		},
		{
			sender: "user",
			message:
				"Lorem ipsum dolor sit amet consectetur, adipisicing elit. Praesentium, id adipisci sequi veritatis ad aliquid repellendus eius, quisquam laboriosam rerum ea mollitia doloribus inventore quod vero? Sunt, sed! Ea, commodi.",
		},
	];

	return (
		<main className='relative h-[80vh] flex flex-col gap-4 overflow-y-auto custom-scrollbar'>
			{chats.map((blob, index) => (
				<div
					key={index}
					className={`max-w-3/5 rounded-2xl p-4 ${
						blob.sender === "user" ? "bg-black ml-auto" : "bg-secondary"
					}`}
				>
					<p>{blob.message}</p>
				</div>
			))}
			<form onSubmit={handleSubmit} className='relative py-12'>
				<input className='fixed bottom-8 mx-auto w-3/5 h-24 bg-white rounded-full px-6' />
			</form>
		</main>
	);
};

export default Home;
