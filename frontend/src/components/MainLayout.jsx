import Header from "./Header";

const MainLayout = ({ children }) => {
	return (
		<div className='text-text-primary'>
			<Header />
			<main className='flex-grow mx-auto py-16 px-64 bg-background'>{children}</main>
		</div>
	);
};

export default MainLayout;
