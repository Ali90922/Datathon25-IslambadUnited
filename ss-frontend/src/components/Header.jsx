import { Link } from "react-router-dom";

const Header = () => {
	return (
		<nav className='bg-linear-to-t from-primary to-background'>
			<ul className='flex items-center justify-center'>
				<Link to='/' className='text-3xl font-extralight p-4'>
					Substance Sense
				</Link>
			</ul>
		</nav>
	);
};

export default Header;
