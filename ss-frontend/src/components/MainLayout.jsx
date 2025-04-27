import Header from "./Header";

const MainLayout = ({ children }) => {
	return (
		<div className='min-h-screen flex flex-col'>
			<Header />
			<main className='flex-grow bg-linear-to-b from-background to-primary pb-32 pt-2'>
				{children}
			</main>
		</div>
	);
};

export default MainLayout;
