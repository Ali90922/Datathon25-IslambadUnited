import { Link } from "react-router-dom";

const Header = () => {
	return (
		<nav className='bg-primary'>
			<ul className='flex items-center justify-center'>
				<Link to='/' className='text-2xl font-bold p-4 hover:bg-secondary'>
					Substance Sense
				</Link>
			</ul>
		</nav>
	);
};

export default Header;
