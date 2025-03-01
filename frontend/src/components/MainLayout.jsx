import Header from "./Header";

const MainLayout = ({ children }) => {
	return (
		<div className='text-text-primary'>
			<Header />
			<main className='flex-grow mx-auto p-8 sm:px-16 md:px-32 lg:px-64 bg-[var(--color-background)]'>
				{children}
			</main>
		</div>
	);
};

export default MainLayout;
